import copy

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