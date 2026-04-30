def copy_A(A):
    """Tạo một bản sao của ma trận để tránh thay đổi dữ liệu gốc."""
    return [row[:] for row in A]

def determinant(A):
    n = len(A)
    # Sao chép ma trận làm việc để không ảnh hưởng đến ma trận truyền vào
    mat = copy_A(A)
    swaps_count = 0  # Biến đếm số lần hoán vị dòng (để đổi dấu định thức)

    for i in range(n):
        # Mặc định phần tử chốt (pivot) là phần tử trên đường chéo chính
        pivot_val = mat[i][i]
        pivot_row = i

        # Chiến lược chọn phần tử chốt một phần (Partial Pivoting)
        # Tìm dòng có giá trị tuyệt đối lớn nhất ở cột hiện tại để giảm sai số
        for j in range(i + 1, n):
            if abs(mat[j][i]) > abs(pivot_val):
                pivot_val = mat[j][i]
                pivot_row = j

        # Nếu phần tử chốt lớn nhất vẫn xấp xỉ 0, ma trận suy biến (định thức = 0)
        if abs(pivot_val) < 1e-12:
            return 0
        
        # Hoán vị dòng hiện tại với dòng chứa phần tử chốt lớn nhất
        if pivot_row != i:
            mat[i], mat[pivot_row] = mat[pivot_row], mat[i]
            swaps_count += 1

        # Quá trình khử Gauss để đưa ma trận về dạng tam giác trên
        for j in range(i + 1, n):
            # Tính hệ số triệt tiêu
            factor = mat[j][i] / pivot_val
            
            # Cập nhật các phần tử còn lại trên dòng j
            for k in range(i, n):
                mat[j][k] -= factor * mat[i][k]
    
    # Định thức của ma trận tam giác bằng tích các phần tử trên đường chéo chính
    det = 1.0
    for i in range(n):
        det *= mat[i][i]
    
    # Điều chỉnh dấu của định thức dựa trên số lần hoán vị dòng ((-1)^k)
    if swaps_count % 2 == 1:
        det = -det
        
    return det