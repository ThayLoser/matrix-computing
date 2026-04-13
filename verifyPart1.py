import numpy as np

"""
Note: Đọc kỹ code từng phần rồi copy hàm kiểm thử của mình vào từng phần mà mình làm
      Các ma trận phải là list của các list
"""

# Hàm kiểm thử tam giác trên (Câu 1)
def verify_tri_solution(U):
    U_np = np.array(U, dtype=float)
    is_upper_tri = np.allclose(U_np, np.triu(U_np))

    if is_upper_tri:
        print("Ma trận đã ở dạng tam giác trên.")
        return True
    else:
        print("Ma trận chưa phải tam giác trên.")
        return False        

# Hàm kiểm thử kết quả Gauss (Câu 2)
def verify_solution(A, x, b):

    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    x_custom = np.array(x, dtype=float)
    
    try:
        x_numpy = np.linalg.solve(A_np, b_np)
        is_match = np.allclose(x_custom, x_numpy)
        
        b_calculated = np.dot(A_np, x_custom)
        is_b_match = np.allclose(b_calculated, b_np)
        
        if is_match and is_b_match:
            print("Kết quả của bạn là đúng.")
            return True
        else:
            print("Kết quả của bạn là sai.")
            print(f"Kết quả NumPy: {x_custom}")
            return False
        
    except np.linalg.LinAlgError as e:
        print(f"Lỗi NumPy: {e}")
        
# Hàm kiểm thử định thức (Câu 3)
def verify_determinant_solution(A, x):
    A_np = np.array(A, dtype=float)
    
    try:
        x_np = np.linalg.det(A_np)

        if np.isclose(x, x_np):
            print("Định thức của bạn là đúng.")
            return True
        else:
            print("Định thức của bạn là sai.")
            return False
    except np.linalg.LinAlgError as e:
        print(f"Lỗi NumPy: {e}")    

# Hàm kiểm thử nghịch đảo (Câu 4)
def verify_inverse_solution(A, x):
    A_np = np.array(A, dtype=float)
    x_np = np.array(x, dtype=float)

    try:
        A_numpy = np.linalg.inv(A_np)
        
        is_match = np.allclose(x_np, A_numpy)
         
        I_test = np.allclose(np.dot(A_np, x_np), np.eye(A_np.shape[0]))
        
        if is_match and I_test:
            print("Kết quả của bạn là đúng.")
            return True
        else:
            print("Kết quả của bạn là sai.")    
            print(f"Kết quả NumPy: {A_numpy}")
            return False
        
    except np.linalg.LinAlgError as e:
        print(f"Lỗi NumPy: {e}")

# Hàm kiểm thử hạng (Câu 5.1)
def verify_rank_solution(A, x):
    A_np = np.array(A, dtype=float)
    
    try:
        x_np = np.linalg.matrix_rank(A_np)
        
        if x == x_np:
            print("Kết quả của bạn là đúng.")
            return True
        else:
            print("Kết quả của bạn là sai.")
            print(f"Kết quả NumPy: {x_np}")
            return False
        
    except np.linalg.LinAlgError as e:
        print(f"Lỗi NumPy: {e}")
        
# Hàm kiểm thử cơ sở của không gian cột (Câu 5.2)
def verify_column_space(A, x):
    A_np = np.array(A, dtype=float)
    
    try:
        rank_A = np.linalg.matrix_rank(A_np)
        
        if len(x) != rank_A:
            print("Sai số chiều không gian cột.")
            print(f"Kết quả NumPy: {rank_A}")
            return False
        
        x_np = np.array(x, dtype=float).T
        A_combined = np.column_stack((A_np, x_np))
        A_combined_rank = np.linalg.matrix_rank(A_combined)
        
        if A_combined_rank == rank_A:
            print("Không gian cột chính xác.")
            return True
        else:
            print("Tập vector không sinh ra không gian cột của A.")
            return False
    except np.linalg.LinAlgError as e:
        print(f"Lỗi NumPy: {e}")

# Hàm kiểm thử cơ sở của không gian dòng (Câu 5.3)
def verify_row_space(A, x):
    A_np = np.array(A, dtype=float)
    
    try:
        rank_A = np.linalg.matrix_rank(A_np)
        
        if len(x) != rank_A:
            print("Sai số chiều không gian dòng.")
            print(f"Kết quả NumPy: {rank_A}")
            return False
        
        A_T = A_np.T
        x_np = np.array(x, dtype=float).T
        A_combined = np.column_stack((A_T, x_np))
        A_combined_rank = np.linalg.matrix_rank(A_combined)
        
        if A_combined_rank == rank_A:
            print("Không gian dòng chính xác.")
            return True
        else:
            print("Tập vector không sinh ra không gian dòng của A.")
            return False
    except np.linalg.LinAlgError as e:
        print(f"Lỗi NumPy: {e}")        

# Hàm kiểm thử cơ sở của không gian nghiệm (Câu 5.4)
def verify_null_space(A, x):
    A_np = np.array(A, dtype=float)
    A_cols = A_np.shape[1]
    rank_A = np.linalg.matrix_rank(A_np)
    
    dim = A_cols - rank_A
    if len(x) != dim:
        print("Sai số chiều.")
        print(f"Số chiều NumPy: {dim}")
        return False
    
    for v in x:
        v_np = np.array(v, dtype=float)
        if not np.allclose(np.dot(A_np, v_np), 0):
            print("Vector trong cơ sở không thỏa mãn Ax = 0.")
            return False
    
    print("Không gian nghiệm chính xác")
    return True

# 5 Test case (tạo bởi AI)
if __name__ == "__main__":
    print("="*50)
    print("BẮT ĐẦU CHẠY 5 TEST CASES CHO TỪNG HÀM KIỂM THỬ")
    print("="*50)

    # ---------------------------------------------------------
    print("\n[1] TEST HÀM TAM GIÁC TRÊN (verify_tri_solution)")
    # 1. Tam giác trên chuẩn 3x3
    verify_tri_solution([[1, 2, 3], [0, 4, 5], [0, 0, 6]]) 
    # 2. Ma trận đơn vị (cũng là tam giác trên)
    verify_tri_solution([[1, 0], [0, 1]]) 
    # 3. Ma trận không phải tam giác trên (Sẽ báo sai)
    verify_tri_solution([[1, 2], [3, 4]]) 
    # 4. Tam giác trên 4x4
    verify_tri_solution([[2, 1, 1, 1], [0, 3, 1, 1], [0, 0, 4, 1], [0, 0, 0, 5]])
    # 5. Ma trận toàn số 0 (hợp lệ tam giác trên)
    verify_tri_solution([[0, 0], [0, 0]])

    # ---------------------------------------------------------
    print("\n[2] TEST HÀM GIẢI HỆ PHƯƠNG TRÌNH (verify_solution)")
    # 1. Hệ 2x2 chuẩn: x + 2y = 5, 3x + 4y = 11 => x=1, y=2
    verify_solution([[1, 2], [3, 4]], [1, 2], [5, 11])
    # 2. Hệ 3x3 chuẩn
    A_sys = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b_sys = [8, -11, -3]
    x_sys = [2, 3, -1]
    verify_solution(A_sys, x_sys, b_sys)
    # 3. Đưa vào nghiệm sai (Sẽ báo sai)
    verify_solution([[1, 2], [3, 4]], [0, 0], [5, 11])
    # 4. Hệ với ma trận đường chéo
    verify_solution([[5, 0], [0, 2]], [2, 3], [10, 6])
    # 5. Ma trận suy biến - không thể giải duy nhất (Sẽ bắt lỗi LinAlgError)
    verify_solution([[1, 1], [2, 2]], [0, 0], [3, 6])

    # ---------------------------------------------------------
    print("\n[3] TEST HÀM ĐỊNH THỨC (verify_determinant_solution)")
    # 1. Ma trận 2x2: det = 1*4 - 2*3 = -2
    verify_determinant_solution([[1, 2], [3, 4]], -2)
    # 2. Ma trận đơn vị: det = 1
    verify_determinant_solution([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 1)
    # 3. Ma trận suy biến (có 2 dòng tỉ lệ): det = 0
    verify_determinant_solution([[1, 2], [2, 4]], 0)
    # 4. Tính sai định thức (Sẽ báo sai)
    verify_determinant_solution([[1, 2], [3, 4]], 100)
    # 5. Ma trận tam giác (det = tích đường chéo = 24)
    verify_determinant_solution([[2, 5, 7], [0, 3, 9], [0, 0, 4]], 24)

    # ---------------------------------------------------------
    print("\n[4] TEST HÀM NGHỊCH ĐẢO (verify_inverse_solution)")
    # 1. Ma trận 2x2 chuẩn
    verify_inverse_solution([[1, 2], [3, 4]], [[-2.0, 1.0], [1.5, -0.5]])
    # 2. Ma trận đơn vị (nghịch đảo là chính nó)
    verify_inverse_solution([[1, 0], [0, 1]], [[1, 0], [0, 1]])
    # 3. Đưa vào kết quả nghịch đảo sai (Sẽ báo sai)
    verify_inverse_solution([[1, 2], [3, 4]], [[1, 1], [1, 1]])
    # 4. Ma trận đường chéo
    verify_inverse_solution([[2, 0], [0, 5]], [[0.5, 0], [0, 0.2]])
    # 5. Ma trận không khả nghịch (Sẽ bắt lỗi LinAlgError)
    verify_inverse_solution([[1, 1], [1, 1]], [[0, 0], [0, 0]])

    # ---------------------------------------------------------
    print("\n[5.1] TEST HÀM TÌM HẠNG (verify_rank_solution)")
    # 1. Ma trận 3x3 độc lập tuyến tính -> Hạng 3
    verify_rank_solution([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 3)
    # 2. Ma trận 3x3 có dòng cuối là tổng 2 dòng đầu -> Hạng 2
    verify_rank_solution([[1, 2, 3], [4, 5, 6], [5, 7, 9]], 2)
    # 3. Ma trận zero -> Hạng 0
    verify_rank_solution([[0, 0], [0, 0]], 0)
    # 4. Cung cấp hạng sai (Sẽ báo sai)
    verify_rank_solution([[1, 2], [3, 4]], 1)
    # 5. Ma trận chữ nhật 2x3 có hạng 2
    verify_rank_solution([[1, 2, 3], [0, 1, 4]], 2)

    # ---------------------------------------------------------
    print("\n[5.2] TEST KHÔNG GIAN CỘT (verify_column_space)")
    # 1. Hạng 2, cơ sở chọn đúng 2 cột đầu
    A_col = [[1, 2, 3], [4, 5, 6], [5, 7, 9]]
    verify_column_space(A_col, [[1, 4, 5], [2, 5, 7]])
    # 2. Ma trận đơn vị (cơ sở là các vector đơn vị)
    verify_column_space([[1, 0], [0, 1]], [[1, 0], [0, 1]])
    # 3. Báo thiếu vector cơ sở (Sẽ báo sai số chiều)
    verify_column_space([[1, 2], [3, 4]], [[1, 3]])
    # 4. Đưa vector không liên quan (Sẽ báo sai tập sinh)
    verify_column_space([[1, 0], [0, 1]], [[1, 1], [99, 99]])
    # 5. Ma trận chữ nhật 2x3, hạng 2
    verify_column_space([[1, 0, 2], [0, 1, 3]], [[1, 0], [0, 1]])

    # ---------------------------------------------------------
    print("\n[5.3] TEST KHÔNG GIAN DÒNG (verify_row_space)")
    # 1. Hạng 2, cơ sở chọn đúng 2 dòng đầu
    A_row = [[1, 2, 3], [4, 5, 6], [5, 7, 9]]
    verify_row_space(A_row, [[1, 2, 3], [4, 5, 6]])
    # 2. Ma trận đơn vị 
    verify_row_space([[1, 0], [0, 1]], [[1, 0], [0, 1]])
    # 3. Báo thừa vector cơ sở (Sẽ báo sai số chiều)
    verify_row_space(A_row, [[1, 2, 3], [4, 5, 6], [5, 7, 9]])
    # 4. Đưa vector sai (Sẽ báo sai tập sinh)
    verify_row_space([[1, 2], [3, 4]], [[1, 1], [0, 0]])
    # 5. Dùng vector cơ sở đã được rút gọn (RREF)
    verify_row_space([[1, 2], [2, 4]], [[1, 2]])

    # ---------------------------------------------------------
    print("\n[5.4] TEST KHÔNG GIAN NGHIỆM (verify_null_space)")
    # 1. Ma trận khả nghịch -> Không gian nghiệm chỉ có vector 0 (chiều = 0)
    verify_null_space([[1, 2], [3, 4]], [])
    # 2. Ma trận [1, -1] (x - y = 0 -> x = y). Cơ sở là [1, 1]
    verify_null_space([[1, -1]], [[1, 1]])
    # 3. Hệ phương trình 2x3, hạng 2, chiều KG nghiệm = 3 - 2 = 1
    # x + z = 0, y = 0 => vector [-1, 0, 1]
    verify_null_space([[1, 0, 1], [0, 1, 0]], [[-1, 0, 1]])
    # 4. Sai số chiều không gian nghiệm
    verify_null_space([[1, -1]], [[1, 1], [2, 2]])
    # 5. Vector cung cấp không thỏa mãn Ax = 0
    verify_null_space([[1, -1]], [[1, 0]])