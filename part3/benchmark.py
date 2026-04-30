import time
import numpy as np
import warnings
import json
from solvers import gauss_elimination, lu_decomposition, gauss_seidel, qr_householder_solve

# Tắt các cảnh báo tràn số (overflow) của numpy khi in ra màn hình
warnings.filterwarnings('ignore')

# Các hàm sinh ma trận test
def generate_random_system(n):
    """Sinh ma trận ngẫu nhiên (Hệ có nghiệm duy nhất)."""
    A = np.random.randn(n, n)
    b = np.random.randn(n)
    return A, b

def generate_spd_system(n):
    """Sinh ma trận Đối xứng Xác định dương (SPD - Well-conditioned)."""
    A = np.random.rand(n, n)
    A = (A + A.T) / 2.0 + n * np.eye(n) # Đối xứng hóa và làm đường chéo lớn lên
    b = np.random.rand(n)
    return A, b

def generate_hilbert_system(n):
    """Sinh ma trận Hilbert (Ill-conditioned - nhạy cảm với sai số)."""
    I, J = np.ogrid[0:n, 0:n]
    A = 1.0 / (I + J + 1.0)
    # Sinh nghiệm chuẩn toàn số 1 để tính b
    x_exact = np.ones(n)
    b = A @ x_exact
    return A, b

# Hàm chạy thực nghiệm
def run_evaluation(solver_func, A, b, num_runs=5):
    """
    Chạy thuật toán 5 lần, lấy trung bình thời gian và tính sai số L2.
    """
    times = []
    x_hat = None
    
    for _ in range(num_runs):
        A_copy = A.copy()
        b_copy = b.copy()
        
        start_time = time.perf_counter()
        try:
            # Chạy thuật toán
            x_hat = solver_func(A_copy, b_copy)
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        except Exception as e:
            return None, f"Lỗi: {str(e)}"
            
    # Tính thời gian trung bình
    avg_time = sum(times) / num_runs
    
    # Tính sai số tương đối L2: ||A*x_hat - b|| / ||b||
    b_hat = A @ x_hat
    error = np.linalg.norm(b_hat - b) / np.linalg.norm(b)
    
    # Kiểm tra phân kỳ (tràn số) cho Gauss-Seidel
    if np.isnan(error) or error > 1.0:
        return avg_time, "Phân kỳ / Tràn số (> 1.0)"
        
    return avg_time, f"{error:.2e}"


# Kịch bản chạy chính
def main():
    sizes = [50, 100, 200, 500, 1000]
    
    scenarios = {
        "Ngẫu nhiên bình thường": generate_random_system,
        "SPD (Xác định dương) - Well-conditioned": generate_spd_system,
        "Hilbert - Ill-conditioned": generate_hilbert_system
    }
    
    solvers = {
        "Gauss": gauss_elimination,
        "LU": lu_decomposition,
        "QR": qr_householder_solve,
        "Gauss-Seidel": gauss_seidel
    }

    print("  BẮT ĐẦU CHẠY THỰC NGHIỆM ĐÁNH GIÁ HIỆU NĂNG VÀ ỔN ĐỊNH SỐ...")
    
    json_data = {
        "sizes": sizes,
        "results": {}
    }
    
    for scenario_name, generator in scenarios.items():
        print(f"\n{'='*70}")
        print(f"  BÀI TEST: MA TRẬN {scenario_name.upper()}")
        print(f"{'='*70}")
        
        # Khởi tạo dict trống cho kịch bản này
        json_data["results"][scenario_name] = {
            "Gauss": {"times": [], "errors": []},
            "LU": {"times": [], "errors": []},
            "QR": {"times": [], "errors": []},
            "Gauss-Seidel": {"times": [], "errors": []}
        }
        
        for n in sizes:
            print(f"\n--- Kích thước N = {n} ---")
            # Khởi tạo ma trận cho kích thước N
            A, b = generator(n)
            
            for solver_name, solver_func in solvers.items():
                time_taken, error_status = run_evaluation(solver_func, A, b, num_runs=5)
                
                json_data["results"][scenario_name][solver_name]["times"].append(time_taken)
                json_data["results"][scenario_name][solver_name]["errors"].append(error_status)
                
                if time_taken is None:
                    # Trường hợp thuật toán văng lỗi
                    print(f"  {solver_name:<15} | {error_status}")
                else:
                    print(f"  {solver_name:<15} | Thời gian (avg 5 lần): {time_taken:.4f}s | Sai số L2: {error_status}")
    
    print("\n" + "="*70)
    print("Đang lưu dữ liệu ra file JSON...")
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
        
    print("Đã lưu thành công file: benchmark_results.json")
if __name__ == "__main__":
    main()