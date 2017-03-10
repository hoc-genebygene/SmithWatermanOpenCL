#define GAP_START_PENALTY -8
#define GAP_EXTEND_PENALTY -1

//kernel void calc_fmat_row(global long * f_mat_prev_row, global long * h_mat_prev_row, global long * f_mat_row) {
//    const int id = get_global_id(0);
//
//    f_mat_row[id] = max(f_mat_prev_row[id], h_mat_prev_row[id] + GAP_START_PENALTY) + GAP_EXTEND_PENALTY;
//}
//
//long pow2(long pow) {
//    return 1 << pow;
//}
//
//kernel void upsweep(global long * padded_row, const long depth) {
//    size_t z = get_global_id(0) * pow2(depth + 1);
//    long left_elem = padded_row[z + pow2(depth) - 1];
//    long right_elem = padded_row[z + pow2(depth + 1) - 1] - (pow2(depth) * GAP_EXTEND_PENALTY);
//    padded_row[z + pow2(depth + 1) - 1] = max(left_elem, right_elem);
//}
//
//kernel void downsweep(global long * padded_row, const long depth) {
//    size_t z = get_global_id(0) * pow2(depth+1);
//    long temp = padded_row[z + pow2(depth) - 1];
//    padded_row[z + pow2(depth) - 1] = padded_row[z + pow2(depth+1) - 1];
//    long left_elem = temp;
//    long right_elem = padded_row[z + pow2(depth+1) - 1];
//    padded_row[z + pow2(depth+1) - 1] = max(left_elem, right_elem) + (pow2(depth) * GAP_EXTEND_PENALTY);
//}

kernel void f_mat_row_kernel(global int * f_mat_prev_row, global int * h_mat_prev_row, global int * f_mat_row) {
	const int id = get_global_id(0);

    if (id == 0) {
        return;
    }

	f_mat_row[id] = max(f_mat_prev_row[id], h_mat_prev_row[id] + GAP_START_PENALTY) + GAP_EXTEND_PENALTY;
}

kernel void h_hat_mat_row_kernel(global int * h_mat_prev_row, global int * subs_score_row, global int * f_mat_row_buffer, global int * h_hat_mat_row_buffer) {
    const int id = get_global_id(0);

    if (id == 0) {
        return;
    }

    h_hat_mat_row_buffer[id] = max(max(h_mat_prev_row[id-1] + subs_score_row[id], f_mat_row_buffer[id]), 0);
}

kernel void h_mat_row_kernel(global int * h_hat_mat_row_buffer, global int * padded_row_buffer, global int * h_mat_row_buffer) {
    const int id = get_global_id(0);

    if (id == 0) {
        return;
    }

    h_mat_row_buffer[id] = max(h_hat_mat_row_buffer[id], padded_row_buffer[id] + GAP_START_PENALTY);
}

int pow2(int pow) {
	return 1 << pow;
}

kernel void zero(global int * row) {
    row[get_global_id(0)] = 0;
}

kernel void upsweep(global int * padded_row, const int depth) {
	size_t z = get_global_id(0) * pow2(depth + 1);
	int left_elem = padded_row[z + pow2(depth) - 1];
	int right_elem = padded_row[z + pow2(depth + 1) - 1] - (pow2(depth) * GAP_EXTEND_PENALTY);
	padded_row[z + pow2(depth + 1) - 1] = max(left_elem, right_elem);
}

kernel void downsweep(global int * padded_row, const int depth) {
	size_t z = get_global_id(0) * pow2(depth + 1);
	int temp = padded_row[z + pow2(depth) - 1];
	padded_row[z + pow2(depth) - 1] = padded_row[z + pow2(depth + 1) - 1];
	int left_elem = temp;
	int right_elem = padded_row[z + pow2(depth + 1) - 1];
	padded_row[z + pow2(depth + 1) - 1] = max(left_elem, right_elem) + (pow2(depth) * GAP_EXTEND_PENALTY);
}
