import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values, log_values_BC, log_values_DAPG
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def validate_pcb(model, dataset, opts, val_bruteforce_cost=None):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)

    avg_cost = cost.mean()
    if opts.normalize_input_reward:
        cost_valid = cost[cost != opts.penalty_per_node * opts.graph_size / (opts.graph_size * 1000)]
        avg_cost_over_valid_routes = cost_valid.mean()
        success_rate = cost_valid.shape[0] / cost.shape[0]
        if opts.eval_only:
            val_bruteforce_cost = torch.tensor(val_bruteforce_cost, dtype=torch.float)
            bruteforce_cost_valid = val_bruteforce_cost[cost != opts.penalty_per_node * opts.graph_size / (opts.graph_size * 1000)]
            comparison = (cost_valid - bruteforce_cost_valid) / bruteforce_cost_valid
            print('Validation BruteForce cost: {}'.format(bruteforce_cost_valid.mean()))
            print('Validation Gap: {:2%}'.format(comparison.mean()))

    else:
        cost_valid = cost[cost != opts.penalty_per_node * opts.graph_size]
        avg_cost_over_valid_routes = cost_valid.mean()
        success_rate = cost_valid.shape[0] / cost.shape[0]
        if opts.eval_only:
            val_bruteforce_cost = torch.tensor(val_bruteforce_cost, dtype=torch.float)
            bruteforce_cost_valid = val_bruteforce_cost[cost != opts.penalty_per_node * opts.graph_size]
            comparison = (cost_valid - bruteforce_cost_valid) / bruteforce_cost_valid
            print('Validation BruteForce cost: {}'.format(bruteforce_cost_valid.mean()))
            print('Validation Gap: {:2%}'.format(comparison.mean()))

    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    print('Validation avg_cost_over_valid_routes: {:2f}'.format(avg_cost_over_valid_routes))
    print('Validation success_rate: {:2%}'.format(success_rate))

    return avg_cost, avg_cost_over_valid_routes, success_rate


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()  # equivalent to model.train(False)

    def eval_model_bat(bat):
        with torch.no_grad():  # no_grad normally pair with model.eval()
            cost, _ = model(move_to(bat, opts.device), normalize=opts.normalize_input_reward)  # _ is log likelihood
            # cost: (batch_size,)
        return cost.data.cpu()
    # bat below is a list-like tensor, and each element is a torch.FloatTensor(batch_size, 2)
    # the number of samples in dataset is val_size
    # batches are picked without replacement
    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)
    # return: (val_size,)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    # step is the overall number of batches trained
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)
        # param_group 0 refers to actor

    # Generate NEW training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution,
        normalize=opts.normalize_input_reward))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)
    # training_dataloader: (epoch_size, graph_size, node_dim)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    # When you want HH:MM:SS rather than only secs...
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    # Save checkpoints...
    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.epoch_start + opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )
    if problem.NAME == 'PcbRoute':
        avg_reward, avg_reward_over_valid_routes, success_rate = validate_pcb(model, val_dataset, opts)
        if not opts.no_tensorboard:
            tb_logger.log_value('val_avg_reward', avg_reward, step)
            tb_logger.log_value('val_avg_reward_over_valid_routes', avg_reward_over_valid_routes, step)
            tb_logger.log_value('val_success_rate', success_rate, step)
    else:
        avg_reward = validate(model, val_dataset, opts)
        if not opts.no_tensorboard:
            tb_logger.log_value('val_avg_reward', avg_reward, step)



    # update baseline...
    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()

    if problem.NAME == 'PcbRoute':
        return avg_reward, avg_reward_over_valid_routes, success_rate
    else:
        return avg_reward


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    # batch: (batch_size, graph_size, node_dim)
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x, normalize=opts.normalize_input_reward)
    # (batch_size,), (batch_size,)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    # Introduction to gradient clipping:
    # https://www.zhihu.com/question/29873016
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)


def train_BC_epoch(model, optimizer, baseline, epoch, val_dataset, problem, tb_logger, opts):
    print("Pre-train epoch {} with behavior cloning, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size_BC // opts.batch_size_BC)
    # step is the overall number of batches trained
    start_time = time.time()

    # Generate NEW training data for each epoch
    dataset_BC, demos, demos_reward = problem.make_dataset_BC(filename=opts.BC_demos_path,
                                                              num_samples=opts.epoch_size,
                                                              offset=step * opts.batch_size,
                                                              normalize=opts.normalize_input_reward)
    training_dataset = baseline.wrap_dataset_BC(dataset_BC, demos, demos_reward)
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)
    # training_dataloader: not only (epoch_size, graph_size, node_dim), as well as baseline eval, demos, demo_rewards

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch_BC(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    # When you want HH:MM:SS rather than only secs...
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    # Save checkpoints...
    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.epoch_start + opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )
    if problem.NAME == 'PcbRoute':
        avg_reward, avg_reward_over_valid_routes, success_rate = validate_pcb(model, val_dataset, opts)
        if not opts.no_tensorboard:
            tb_logger.log_value('val_avg_reward', avg_reward, step)
            tb_logger.log_value('val_avg_reward_over_valid_routes', avg_reward_over_valid_routes, step)
            tb_logger.log_value('val_success_rate', success_rate, step)
    else:
        avg_reward = validate(model, val_dataset, opts)
        if not opts.no_tensorboard:
            tb_logger.log_value('val_avg_reward', avg_reward, step)



    # update baseline...
    baseline.epoch_callback(model, epoch)

    if problem.NAME == 'PcbRoute':
        return avg_reward, avg_reward_over_valid_routes, success_rate
    else:
        return avg_reward


def train_batch_BC(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    # batch: (batch_size, graph_size, node_dim)
    x, demo, demo_rewards = baseline.unwrap_batch_BC(batch)
    # print('x: {}'.format(x))
    # print('bl_val: {}'.format(bl_val))
    # print('demo: {}'.format(demo))
    # print('demo_rewards: {}'.format(demo_rewards))
    x = move_to(x, opts.device)
    # bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    action_log_p = model(x, is_BC=True, normalize=opts.normalize_input_reward)
    # (batch_size,), (batch_size,)
    demo_log_p = model.get_demo_logp(x, demo)
    # log_likelihood
    # # Evaluate baseline, get baseline loss if any (only for critic)
    # bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    # reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    # loss = reinforce_loss + bl_loss

    KDdiv = torch.nn.KLDivLoss(reduction='batchmean')
    loss = KDdiv(demo_log_p, torch.exp(action_log_p))

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    # grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    # Introduction to gradient clipping:
    # https://www.zhihu.com/question/29873016
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values_BC(loss, epoch, batch_id, step, tb_logger, opts)


def train_DAPG_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    # step is the overall number of batches trained
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)
        # param_group 0 refers to actor
    actor_batch_samples = int(opts.batch_size_DAPG * opts.DAPG_actor_ratio)
    batch_num = int(opts.epoch_size_DAPG / opts.batch_size_DAPG)
    actor_epoch_samples = batch_num * actor_batch_samples
    demo_batch_samples = opts.batch_size_DAPG - actor_batch_samples
    demo_epoch_samples = demo_batch_samples * batch_num

    # Generate NEW training data for each epoch
    training_dataset_actor = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=actor_epoch_samples, distribution=opts.data_distribution,
        normalize=opts.normalize_input_reward))
    training_dataloader_actor = DataLoader(training_dataset_actor, batch_size=actor_batch_samples, num_workers=1)

    dataset_DAPG_demos, demos, demos_reward = problem.make_dataset_DAPG_demos(filename=opts.BC_demos_path,
                                                                              num_samples=demo_epoch_samples,
                                                                              normalize=opts.normalize_input_reward)
    training_dataset_demos = baseline.wrap_dataset_BC(dataset_DAPG_demos, demos, demos_reward)
    training_dataloader_demos = DataLoader(training_dataset_demos, batch_size=demo_batch_samples, num_workers=1)
    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, (batch_actor, batch_demos) in enumerate(tqdm(zip(training_dataloader_actor, training_dataloader_demos), disable=opts.no_progress_bar)):

        train_DAPG_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch_actor,
            batch_demos,
            tb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    # When you want HH:MM:SS rather than only secs...
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    # Save checkpoints...
    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.epoch_start + opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )
    if problem.NAME == 'PcbRoute':
        avg_reward, avg_reward_over_valid_routes, success_rate = validate_pcb(model, val_dataset, opts)
        if not opts.no_tensorboard:
            tb_logger.log_value('val_avg_reward', avg_reward, step)
            tb_logger.log_value('val_avg_reward_over_valid_routes', avg_reward_over_valid_routes, step)
            tb_logger.log_value('val_success_rate', success_rate, step)
    else:
        avg_reward = validate(model, val_dataset, opts)
        if not opts.no_tensorboard:
            tb_logger.log_value('val_avg_reward', avg_reward, step)



    # update baseline...
    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()

    if problem.NAME == 'PcbRoute':
        return avg_reward, avg_reward_over_valid_routes, success_rate
    else:
        return avg_reward


def train_DAPG_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch_actor,
        batch_demos,
        tb_logger,
        opts
):
    # batch: (batch_size, graph_size, node_dim)
    x, bl_val = baseline.unwrap_batch(batch_actor)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    x_demos, action_demos, demo_rewards = baseline.unwrap_batch_BC(batch_demos)

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood_actor = model(x, normalize=opts.normalize_input_reward)
    # (batch_size,), (batch_size,)

    _, log_likelihood_demo = model.get_demo_logp(x_demos, action_demos, on_DAPG=True)

    best_reward_from_actions = opts.DAPG_lam_0 * opts.DAPG_lam_1 ** epoch * torch.min(cost - bl_val)
    # Calculate loss
    value_function = torch.cat([(cost - bl_val), best_reward_from_actions.expand(len(x_demos))])
    # print('log_likelihood_actor', log_likelihood_actor)
    # print('log_likelihood_demo', log_likelihood_demo)
    ll = torch.cat([log_likelihood_actor, log_likelihood_demo])
    loss = (value_function * ll).mean()

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    # Introduction to gradient clipping:
    # https://www.zhihu.com/question/29873016
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values_DAPG(cost, grad_norms, epoch, batch_id, step,
                        log_likelihood_actor, log_likelihood_demo, loss, tb_logger, opts)
