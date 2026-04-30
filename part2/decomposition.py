import copy
epsilon = 1e-10

# code rank_and_basis
def to_ref(A: list[list]) -> list[list]:
    B = copy.deepcopy(A)
    epsilon = 1e-12
    m = len(B)
    n = len(B[0])
    pr =pc = 0
   
    while pr < m and pc < n:
        # Hoán với dòng có |A_ik|max
        p = pr
        for i in range(pr+1, m):
            if abs(B[i][pc]) > abs(B[p][pc]): 
                p=i

        if p != pr:
            B[pr], B[p] = B[p], B[pr]

        # không có pivot
        if abs(B[pr][pc]) < epsilon: 
            pc += 1
            continue

        # Khử các dòng dưới
        for i in range(pr+1, m):
            factor = B[i][pc]/B[pr][pc]
            for j in range(pc,n):
                B[i][j] -= factor*B[pr][j]

        pr+=1
        pc+=1
    return B
def to_rref(A: list[list]) -> list[list]:
    B = to_ref(A)

    epsilon = 1e-12
    m = len(B)
    n = len(B[0])
    pr = m-1
    pc = n-1

    while(pr >=0):
        # Tìm pivot
        found = False 
        for i in range(pr,-1, -1):
            if found: break 

            for j in range(pc, -1, -1):
                if abs(B[i][j]) > epsilon:
                    pr, pc = i, j
                    found = True
        
        if found:
            # Leading 1
            for j in range(n-1, pc-1, -1):
                B[pr][j]/=B[pr][pc]

            # Khử các dòng trên
            for i in range(pr-1, -1, -1):
                if abs(B[i][pc]) > epsilon:
                    factor = B[i][pc]
                    for j in range(pc, n):
                        B[i][j] -= factor*B[pr][j] 

        pr -= 1
        pc -= 1
    return B
def rank_and_basis(A: list[list]) -> tuple[int, list[list[float]], list[list[float]], list[list[float]]]:
    rrefA = to_rref(A)

    m = len(A)
    n = len(A[0])
    epsilon = 1e-12

    
    row_basis = []
    col_basis = []
    null_basis = []
    # Tính rank
    rank = 0
    for i in range(m-1, -1, -1):
        if rank > epsilon: break

        for j in range(0,n):
            if abs(rrefA[i][j]) > epsilon:
                rank=i+1
                break
    # Cơ sở dòng
    row_basis = [rrefA[i] for i in range(rank)]
    # Cơ sở cột
    pc = 0
    for i in range(rank):
        # Tìm pivot
        while pc < n and abs(rrefA[i][pc] - 1) > epsilon:
            pc+=1

        if pc >= n or abs(rrefA[i][pc] - 1) > epsilon: break
        # Thêm cột
        col = []
        for i_ in range(m):
            col.append(A[i_][pc])
        col_basis.append(col)

    # Cơ sở nghiệm
    C = [[0] * n for _ in range(n)] 
    pivots = [] # Đánh dấu pivot (các ẩn phụ thuộc)
    for j in range(n): C[j][j] = 1
    for i in range(m):
        pivot = -1
        for j in range(n):
            if abs(rrefA[i][j]) > epsilon:
                pivot = j 
                pivots.append(j)
                break
        if pivot==-1: continue
        for j in range(pivot+1, n):
            C[j][pivot] = -rrefA[i][j]

    # Giữ lại biến tự do
    null_basis = [C[j] for j in range(n) if j not in pivots]

    return rank, col_basis, row_basis, null_basis

# code QR
def inner_product(v1: list, v2: list) -> float:
    if len(v1) != len(v2):
        raise ValueError("Hai vector phải có cùng độ dài")

    return sum(x * y for x, y in zip(v1, v2))

def sum_vector(v1: list, v2: list) -> list:
    if len(v1) != len(v2):
        raise ValueError("Hai vector phải có cùng độ dài")

    return [x + y for x, y in zip(v1, v2)]

def qr_decomposition(A: list[list]) -> tuple[list[list], list[list]]:
    m = len(A) # số hàng của ma trận gốc
    n = len(A[0]) # số cột của ma trận gốc

    if n*m == 0:
        return [], []

    Q = [] 
    u = transpose(A) # chuyển các cột thành hàng, nxm

    for i in range(n):
        v_i = u[i].copy()
        for _v in Q: # trực hóa với các vector trong Q
            _v_len = inner_product(_v, _v)
            if _v_len > epsilon:
                # v_i -= <v_i, _v> / <_v, _v> * _v

                proj = inner_product(v_i, _v) / _v_len

                v_i = sum_vector(
                        v_i, 
                        [-proj * _v[j] for j in range(m)]
                    ) 

        Q.append(v_i)

    # Chuẩn hóa các vector trong Q(nxm)
    for i in range(n):
        v_i = Q[i]
        norm_v_i = sum(x**2 for x in v_i) ** 0.5
        if norm_v_i > epsilon:
            Q[i] = [x / norm_v_i for x in v_i]
        else:
            Q[i] = [0.0] * m

    # Tính R(nxn), mỗi phần tử R[i][j] = <u[j], Q[i]> nếu i <= j
    R = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            R[i][j] = inner_product(u[j], Q[i])

    return transpose(Q), R

def transpose(A: list[list]) -> list[list]:
    n = len(A) # số hàng của ma trận gốc
    m = len(A[0]) # số cột của ma trận gốc
    
    if n*m == 0:
        return []
    
    res = [[A[i][j] for i in range(n)] for j in range(m)] #hoán đổi chỉ số hàng và cột
    return res

def multiply(A: list[list], B: list[list]) -> list[list]:
    """Trả về tích của A và B"""
    if len(A)*len(B) == 0 or len(A[0])*len(B[0]) == 0 or len(A[0]) != len(B):
        return []
    
    m, n = len(A), len(B[0])

    # Khởi tạo ma trận kết quả với kích thước m x n
    result = [[0] * n for _ in range(m)]
    # Tính tích A * B
    for i in range(m):
        for j in range(n):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(len(B)))

    return result

def normalizevector(v: list) -> list:
    """Trả về vector v đã được chuẩn hóa (||v|| = 1)"""
    # Độ dài n của v
    n = sum(x**2 for x in v) ** 0.5
    if n == 0:
        return v
    
    # Chia từng phần tử rồi trả về 
    return [x / n for x in v]

# M: tranpose(A)*A
def eigen_values(M: list[list]) -> list:
    #tính tổng đường chéo 
    def off_diagonal_sum(A):
        return sum(abs(A[i][j]) for i in range(len(A)) for j in range(i))
    #lặp QR
    max_iterations = 1000
    Mk = [row[:] for row in M] #copy lại ma trận gốc
    
    for iteration in range(max_iterations):
        #phân tích QR 
        Q, R = qr_decomposition(Mk)
        #cập nhật Mk+1 = R * Q
        Mk = multiply(R, Q)
        if off_diagonal_sum(Mk) < epsilon:
            break
            
    return sorted([Mk[i][i] for i in range(len(Mk))], reverse = True)
  
def orthoNormalize(vectors: list[list]) -> list[list]:
    """Trả về danh sách các vector đã được trực chuẩn hóa (orthonormalized)"""
    # Sử dụng thuật toán Gram-Schmidt để trực chuẩn hóa
    orthonormalized = []
    for v in vectors:
        w = copy.deepcopy(v)
        for u in orthonormalized:
            l_u = sum(u[i] * u[i] for i in range(len(u)))  # ||u||^2
            if l_u == 0:
                continue

            proj = sum(w[i] * u[i] for i in range(len(w))) / l_u  # (w . u) / ||u||^2
            w = [w[i] - proj * u[i] for i in range(len(w))] # w - proj_u(w)
        orthonormalized.append(normalizevector(w))
    return orthonormalized

def eigen_vectors(M: list[list], lambdas: list) -> list[list]:
    n = len(M)
    V = []
    unique_lambdas = [] # tránh tính toán lại cho các giá trị trùng lặp
    for lam in lambdas:
        if not any(abs(lam - u_lam) < epsilon for u_lam in unique_lambdas):
            unique_lambdas.append(lam)

    for lambda_i in unique_lambdas:
        A = copy.deepcopy(M)
        for i in range(n):
            A[i][i] -= lambda_i

        _, __, ___, null_basis = rank_and_basis(A)
        
        # trực chuẩn hóa từng vector trong null_basis và thêm vào V
        ortho_basis = orthoNormalize(null_basis)
        for v in ortho_basis:
            V.append(v)

    while len(V) < n:
        new_v = None
        for t in range(n):
            w = [0.0] * n
            w[t] = 1.0
            for u in V:
                proj = sum(w[p] * u[p] for p in range(n))
                w = [w[p] - proj * u[p] for p in range(n)]
            nw = sum(x*x for x in w) ** 0.5
            if nw > epsilon:
                new_v = [x / nw for x in w]
                break
        if new_v is not None:
            V.append(new_v)
        else:
            break
    return V

def build_V(v: list) -> list[list]:
   return transpose(v)

def build_S(A: list[list], sigmas: list) -> list[list]: # SIGMA
    """Trả về ma trận S (ma trận đường chéo) chứa các σ_i trên đường chéo chính"""
    m = len(A)  
    n = len(A[0]) 

    S = [[0.0] * n for _ in range(m)] # S_mxn

    for i in range(min(m, n, len(sigmas))):
        S[i][i] = sigmas[i]

    return S

def build_U(A: list[list], V: list[list], S: list[list]) -> list[list]:
    U = []
    m = len(A)
    n = len(A[0])
    num_sigmas = min(m, n)

    for i in range(num_sigmas):
        sigma_i = S[i][i]
        if sigma_i > epsilon:
            v_i = [[row[i]] for row in V]
            Av_i = multiply(A, v_i)
            u_i = [[Av_i[j][0] / sigma_i] for j in range(len(Av_i))]
            U.append(transpose(u_i)[0])

    # Bổ sung thêm các vector trực chuẩn vào U nếu thiếu
    while len(U) < m:
        ortho = [normalizevector(u) for u in U]
        new_u = None
        for t in range(m):
            w = [0.0] * m
            w[t] = 1.0
            for u in ortho:
                proj = sum(w[p] * u[p] for p in range(m))
                w = [w[p] - proj * u[p] for p in range(m)]
            nw = sum(x*x for x in w) ** 0.5
            if nw > epsilon:
                new_u = [x / nw for x in w]
                break
        if new_u is not None:
            U.append(new_u)
        else:
            break

    return transpose(U)
  
def svd(A: list[list]):
    M = multiply(transpose(A), A)
    
    lambdas = eigen_values(M)
    lambdas_pos = [l for l in lambdas if l > epsilon]
    sigmas = [l**0.5 for l in lambdas_pos]
    eigvecs = eigen_vectors(M, lambdas_pos)

    V = build_V(eigvecs)
    S = build_S(A, sigmas)
    U = build_U(A, V, S)
    return U, S, V

class LUDecomposition: 
    @staticmethod
    def create_identity_matrix(size):
        return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    
    @staticmethod
    def copy_matrix(matrix):
        return [row[:] for row in matrix]
    
    def __init__(self,A):
        if not A or not A[0]:
            raise ValueError("Ma trận đầu vào không được rỗng")
        self.m=len(A)
        self.n=len(A[0])
        self.A = self.copy_matrix(A)
        self.P = None
        self.L = None
        self.U = None
        
    def decompose(self):
        m, n = self.m, self.n
        
        # P,L là ma trận vuông mxm
        P = self.create_identity_matrix(m)
        L = self.create_identity_matrix(m)
        
        # U là ma trận mxn
        U=self.copy_matrix(self.A)
        
        # Số bước khử tối đa
        limit=min(m,n)
        
        for i in range(limit):
            pivot_row = i
           #  Tìm phần tử lớn nhất trong cột i, từ hàng i đến hàng m - 1
            max_val = abs(U[i][i])
            for k in range(i + 1, m):
                if abs(U[k][i]) > max_val:
                    max_val = abs(U[k][i])
                    pivot_row = k
                    
            # Nếu toàn bộ cột dưới đường chéo đều là 0, bỏ qua không cần khử
            if max_val == 0:
                continue
            
            # Đổi hàng nếu cần
            if pivot_row != i:
                # Đổi hàng trong U
                U[i], U[pivot_row] = U[pivot_row], U[i]
                # Đổi hàng trong P
                P[i], P[pivot_row] = P[pivot_row], P[i]
                
                # Đổi hàng trong L (chỉ đổi các cột bên trái phần tử đường chéo i)
                for k in range(i):
                    L[i][k], L[pivot_row][k] = L[pivot_row][k], L[i][k]
                    
            # 2. Khử Gauss
            for j in range(i + 1, m):
                factor = U[j][i] / U[i][i]
                L[j][i] = factor
                
                # Cập nhật các phần tử còn lại trên hàng j của U
                for k in range(i, n):
                    U[j][k] -= factor * U[i][k]
                    
        self.P, self.L, self.U = P, L, U
        return P, L, U