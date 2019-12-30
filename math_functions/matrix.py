import numpy as np

A = np.array([[1., 2, 3, 2], [4, 3, 7, 4], [1, 4, 2, 3]])
print(A)


print('np.sum on the axis = [0, 1], with keeping dimension of the root matrix-----------------------------------------')
sum_A_axis0_no_keepdims = np.sum(A, axis=0)
sum_A_axis0_with_keepdims = np.sum(A, axis=0, keepdims=True)

sum_A_axis1_no_keepdims = np.sum(A, axis=1,)
sum_A_axis1_with_keepdims = np.sum(A, axis=1, keepdims=True)


# Norm 2: calculate norm2 of B matrix
B = np.array([[1, 3], [2, 5]])


# Trong khi làm việc với Machine Learning, chúng ta thường xuyên phải so sánh hai ma trận.
# Xem xem liệu chúng có gần giống nhau không. Một cách phổ biến để làm việc này là tính
# bình phương của Frobineous norm của hiệu hai ma trận đó
# Cho hai mảng hai chiều có cùng kích thước A và B. Viết hàm dist_fro tính bình phương
# Frobenious norm của hiệu hai ma trận được mô tả bởi hai mảng đó.