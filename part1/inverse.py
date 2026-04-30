def inverse(A):
    n = len(A)
    matrix = [] #khởi tạo ma trận 
    
    #bước đầu: tạo ma trận (A|I)
    for i in range(n):
        row = []
        #ép kiểu sang float
        for j in range(n):
            row.append(float(A[i][j]))
        #thêm ma trận đơn vị I
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                row.append(0.0)
        matrix.append(row)
    #thực hiện các bước gauss-jordan
    for i in range(n):
        #bước 1:tìm pivoting (tìm dòng có giá trị lớn nhất)
        max_row = i
        for k in range(i+1,n): #duyệt hết các dòng bên dưới
            if abs(matrix[k][i]) > abs(matrix[max_row][i]):
                max_row = k
        #đổi dòng
        temp = matrix[i]
        matrix[i] = matrix[max_row]
        matrix[max_row] = temp
        #kiểm tra tính khả nghịch xem ma trận có suy biến không
        if abs(matrix[i][i]) < 1e-10:
            return None #ma trận không khả nghịch
        #bước 2: chuẩn hóa pivot về 1
        pivot = matrix[i][i]
        for j in range(2*n):
            matrix[i][j] /= pivot
        #bước 3: khử các dòng khác
        for k in range(n):
            if k != i:
                factor = matrix[k][i]
                for j in range(2*n):
                    matrix[k][j] -= factor * matrix[i][j]
    inv_A = []
    for i in range(n):
        row = []
        for j in range(n, 2 * n):
            row.append(matrix[i][j])
        inv_A.append(row)
    return inv_A