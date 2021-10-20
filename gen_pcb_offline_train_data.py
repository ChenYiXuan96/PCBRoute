import json
import copt


def generate_offline_pcb_data(n_lines=5, size=2500, early_exit=0, path='pcb_5_2.5k_validate'):
    result = []
    failed_generation = 0
    for _ in range(size):
        problem = copt.getProblem(n_lines)
        solutions = copt.bruteForce(problem, early_exit)
        if solutions:
            result.append((problem, solutions[0]))
        else:
            failed_generation += 1
            print('Generation failed. Total fail: {}.\n'.format(failed_generation))
    with open(path, 'w') as file:
        json.dump(result, file)


if __name__ == '__main__':
    generate_offline_pcb_data()
