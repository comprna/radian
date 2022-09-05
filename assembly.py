from sklearn.preprocessing import normalize

def assemble_matrices(matrices, step_size):
    # Stack the overlapping matrices together
    stack = stack_matrices(matrices, step_size)
    # Collapse the stack
    return collapse_matrices_stack(stack)

def stack_matrices(read_matrices, step_size):
    # Combine window matrices into a single read matrix
    # (jagged list of lists --> 1 list per timestep)
    stack = []
    t_start = 0
    for batch_matrices in read_matrices:
        for matrix in batch_matrices:
            # Start at the appropriate timestep
            t_curr = t_start

            for dist in matrix:
                # Extend pile with current timestep if it isn't
                # already in there
                if t_curr >= len(stack):
                    stack.append([])
                
                # Add distribution to current timestep
                stack[t_curr].append(dist)

                # Increment current time
                t_curr += 1

            # Once all distributions added, increment t_start by step_size
            t_start += step_size
    return stack

def collapse_matrices_stack(stack):
    global_matrix = []
    for t, dist_list in enumerate(stack):         
        # Combine all distributions at the current timestep
        # Approach 2:
        if len(dist_list) > 1:
            global_matrix.append(sum_normalised_list_l1(dist_list))
        else:
            global_matrix.append(dist_list[0])
    return np.asarray(global_matrix)

def sum_normalised_list_l1(dists):
    result = dists[0]
    for i, dist in enumerate(dists):
        # Already added first item before loop
        if i == 0:
            continue
        np.add(result, dist)
    result = normalize([result], norm="l1")[0]
    return result
