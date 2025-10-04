# 과제 조건
#  1. 행렬 입력 기능
#   -- 사용자로부터 정수 n을 입력받아 n × n 크기의 정방행렬을 행 단위로 입력받을 수 있어야 함
#   -- 입력 데이터는 리스트(2차원 배열)로 저장
#  2. 행렬식을 이용한 역행렬 계산 기능
#   -- 행렬식을 사용하여 주어진 행렬의 역행렬을 계산하는 함수를 구현
#   -- 역행렬이 존재하지 않을 경우(행렬식이 0인 경우) 오류 메시지를 출력
#  3. 가우스-조던 소거법(Gauss-Jordan elimination)을 이용한 역행렬 계산 기능1
#   -- 가우스-조던 소거법을 사용하여 동일한 행렬의 역행렬을 계산하는 함수를 구현
#   -- 행렬식과 마찬가지로 역행렬이 존재하지 않는 경우 예외 처리 포함
#  4. 결과 출력 및 비교 기능
#   -- 두 방법(행렬식, 가우스-조던)으로 계산한 역행렬을 각각 출력
#   -- 두 결과가 동일한지 비교하여 결과 메시지를 출력

# 수행 결과
# 기능 포함 : n×n 입력, 행렬식(Adjugate) 역행렬, 가우스-조던 역행렬,
#           두 결과 출력/동일성 비교, A·A^{-1} 단위행렬 검증, 속도 비교,
#           기본 연산(덧셈·곱셈·전치), 견고한 예외처리, 메뉴 기반 반복 실행

from typing import List, Tuple, Optional
import time

# =========================
# 공통 유틸
# =========================

EPS_DET = 1e-12
EPS_EQ  = 1e-7

def clone_matrix(M: List[List[float]]) -> List[List[float]]:
    return [row[:] for row in M]

def eye(n: int) -> List[List[float]]:
    I = [[0.0]*n for _ in range(n)]
    for i in range(n): I[i][i] = 1.0
    return I

def transpose(M: List[List[float]]) -> List[List[float]]:
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

def pretty(M, name: str = "", eps: float = 1e-12):
    if name:
        print(name)
    n = len(M)
    if n == 0:
        print("[빈 행렬]")
        return

    # 문자열 행렬 생성 + 최대 길이 계산
    str_matrix = []
    max_len = 0
    for i in range(n):
        row_strs = []
        for j in range(len(M[i])):
            x = M[i][j]
            if abs(x) < eps:
                s = "0"
            else:
                xi = round(x)
                if abs(x - xi) < eps:
                    s = f"{xi:d}"
                else:
                    if (abs(x) >= 1e6) or (abs(x) <= 1e-6):
                        s = f"{x:.6e}"
                    else:
                        s = f"{x:.6f}"
            row_strs.append(s)
            if len(s) > max_len:
                max_len = len(s)
        str_matrix.append(row_strs)

    # 셀 폭 계산(좌우 여백 포함)
    inner_pad = 2 
    cell_w = max_len + 2 * inner_pad

    # 행 문자열 만들기
    row_lines = []
    for row in str_matrix:
        row_fmt = "".join(f"{s:^{cell_w}}" for s in row)  # 가운데 정렬
        row_lines.append(row_fmt)

    # 테두리 폭 계산 + 내부에 좌우 1칸 여백 넣음
    inside_w = len(row_lines[0]) + 2

    # 상하/좌우 테두리 + 행 출력
    print("┌" + " " * inside_w + "┐")
    for line in row_lines:
        print("│ " + line + " │")
    print("└" + " " * inside_w + "┘")



def almost_equal(a: float, b: float, eps: float = EPS_EQ) -> bool:
    return abs(a - b) <= eps

def matrices_equal(A: List[List[float]], B: List[List[float]], eps: float = EPS_EQ) -> bool:
    if len(A) != len(B) or len(A[0]) != len(B[0]): return False
    n, m = len(A), len(A[0])
    for i in range(n):
        for j in range(m):
            if not almost_equal(A[i][j], B[i][j], eps): return False
    return True

# =========================
# 입력/검증
# =========================

def read_square_matrix():
    while True:
        try:
            n = int(input("정방행렬 크기 n : ").strip())
            if n <= 0:
                print("n은 양의 정수여야 함.")
                continue
            print(f"{n}×{n} 행을 공백 구분으로 차례대로 입력")
            A = []
            for i in range(n):
                while True:
                    row = input(f"{i+1}행: ").strip().split()
                    if len(row) != n:
                        print(f"→ {n}개 값을 정확히 입력해야함.")
                        continue
                    try:
                        A.append([float(x) for x in row])
                        break
                    except ValueError:
                        print(" → 숫자만 입력.")
            # 여기 추가 ↓
            print("\n[디버깅] 입력 원시 행 : ")
            print(repr(A))
            pretty(A, "\n입력 행렬 A : ")
            return A
        except ValueError:
            print(" → n은 정수.")


# =========================
# 기본 연산
# =========================

def add_matrix(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

def multiply_matrix(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n = len(A)
    return [[sum(A[i][k]*B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

def is_identity(M: List[List[float]], eps: float = EPS_EQ) -> bool:
    n = len(M)
    for i in range(n):
        for j in range(n):
            target = 1.0 if i == j else 0.0
            if abs(M[i][j] - target) > eps:
                return False
    return True

# =========================
# 행렬식/여인자 방식 역행렬
# =========================

def minor_matrix(M: List[List[float]], r: int, c: int) -> List[List[float]]:
    return [row[:c] + row[c+1:] for row in (M[:r] + M[r+1:])]

def det(M: List[List[float]]) -> float:
    n = len(M)
    if n == 1: return M[0][0]
    if n == 2: return M[0][0]*M[1][1] - M[0][1]*M[1][0]
    s = 0.0
    for c in range(n):
        s += ((-1.0)**c) * M[0][c] * det(minor_matrix(M, 0, c))
    return s

def inverse_by_adjugate(M: List[List[float]]) -> Tuple[List[List[float]], float]:
    n = len(M)
    d = det(M)
    if abs(d) <= EPS_DET:
        raise ValueError("역행렬 없음. 행렬식(det) = 0")
    if n == 1: return [[1.0/M[0][0]]], d
    if n == 2:
        inv = [[ M[1][1]/d, -M[0][1]/d],
               [-M[1][0]/d,  M[0][0]/d]]
        return inv, d
    C = [[0.0]*n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            C[r][c] = ((-1.0)**(r+c)) * det(minor_matrix(M, r, c))
    Adj = transpose(C)
    Inv = [[Adj[i][j]/d for j in range(n)] for i in range(n)]
    return Inv, d

# =========================
# 가우스-조던 방식 역행렬
# =========================

def inverse_by_gauss_jordan(M: List[List[float]]) -> List[List[float]]:
    n = len(M)
    A = clone_matrix(M)
    I = eye(n)
    for col in range(n):
        # 부분 피벗팅
        pivot_row = max(range(col, n), key=lambda r: abs(A[r][col]))
        if abs(A[pivot_row][col]) <= EPS_DET:
            raise ValueError("역행렬 없음. 가우스-조던 pivot ≈ 0)")
        if pivot_row != col:
            A[col], A[pivot_row] = A[pivot_row], A[col]
            I[col], I[pivot_row] = I[pivot_row], I[col]
        # 피벗 1로
        pv = A[col][col]
        inv_pv = 1.0 / pv
        A[col] = [x * inv_pv for x in A[col]]
        I[col] = [x * inv_pv for x in I[col]]
        # 타 행 0으로
        for r in range(n):
            if r == col: continue
            factor = A[r][col]
            if abs(factor) <= EPS_DET: continue
            A[r] = [A[r][j] - factor * A[col][j] for j in range(n)]
            I[r] = [I[r][j] - factor * I[col][j] for j in range(n)]
    return I  # 오른쪽 블록이 A^{-1}

# =========================
# 실행/메뉴
# =========================

def compute_and_show_all(A: List[List[float]]) -> None:
    pretty(A, "\n입력 행렬 A : ")

    inv_adj: Optional[List[List[float]]] = None
    inv_gj:  Optional[List[List[float]]] = None

    # 행렬식
    print("\n[행렬식을 이용한 역행렬 계산]")
    t0 = time.time()
    try:
        inv_adj, d = inverse_by_adjugate(A)
        t1 = time.time()
        print(f"det(A) = {d:.8f}  |  시간 : {t1 - t0:.6f}초")
        pretty(inv_adj, "A^{-1} : ")
        # 검증: A·A^{-1} ≈ I
        # 역행렬 계산 뒤
        Ai = multiply_matrix(A, inv_adj)
        pretty(Ai, "검증 A·A^{-1} :")
        print(" → ", "OK" if is_identity(Ai) else "FAIL")

    except ValueError as e:
        t1 = time.time()
        print(f"실패: {e}  |  시간: {t1 - t0:.6f}초")

    # 가우스-조던
    print("\n[가우스-조던 소거법]")
    t2 = time.time()
    try:
        inv_gj = inverse_by_gauss_jordan(A)
        t3 = time.time()
        print(f"시간: {t3 - t2:.6f}초")
        pretty(inv_gj, "A^{-1} (Gauss-Jordan):")
        # 검증: A·A^{-1} ≈ I
        Aj = multiply_matrix(A, inv_gj)
        print("검증(GJ):  A·A^{-1} ≈ I  →", "OK" if is_identity(Aj) else "FAIL")
    except ValueError as e:
        t3 = time.time()
        print(f"실패: {e}  |  시간: {t3 - t2:.6f}초")

    # 두 결과 비교
    print("\n[결과 출력 및 비교]")
    if inv_adj is not None and inv_gj is not None:
        same = matrices_equal(inv_adj, inv_gj, eps=EPS_EQ)
        print("두 방법 결과 동일함" if same else "두 방법 결과가 수치적으로 다름 ")
    else:
        print("두 방법 중 하나 이상 실패로 비교 불가.")

def menu():
    A: Optional[List[List[float]]] = None
    while True:
        print("\n=== 메뉴 ===")
        print("1) 행렬 입력")
        print("2) 행렬식을 이용한 역행렬 계산")
        print("3) 가우스-조던 소거법을 이용한 역행렬 계산")
        print("4) 결과 출력 및 비교")
        print("5) [추가] 행렬 기본연산 - 덧셈/곱셈/전치")
        print("0) 종료")
        sel = input("선택: ").strip()

        if sel == "0":
            print("끝.")
            break

        elif sel == "1":
            A = read_square_matrix()
            pretty(A, "현재 저장된 행렬 A :")

        elif sel == "2":
            if A is None:
                print("먼저 A를 입력해야함.")
                continue
            try:
                inv, d = inverse_by_adjugate(A)
                print(f"det(A) = {d:.8f}")
                pretty(inv, "행렬식을 이용한 역행렬 계산 결과 :")
                print("검증 : ", "OK" if is_identity(multiply_matrix(A, inv)) else "FAIL")
            except ValueError as e:
                print("실패 : ", e)

        elif sel == "3":
            if A is None:
                print("먼저 A를 입력해야함.")
                continue

            try:
                inv = inverse_by_gauss_jordan(A)
                pretty(inv, "가우스-조던 소거법을 이용한 역행렬 계산 결과 : ")
                print("검증 : ", "OK" if is_identity(multiply_matrix(A, inv)) else "FAIL")
            except ValueError as e:
                print("실패 : ", e)

        elif sel == "4":
            if A is None:
                print("먼저 A를 입력해야함.")
                continue
            compute_and_show_all(A)

        elif sel == "5":
            if A is None:
                print("먼저 A를 입력해야함.")
                continue
            print("\n[기본연산 데모]")
            pretty(A, "A :")
            At = transpose(A)
            pretty(At, "A^T :")
            # A + A^T
            try:
                S = add_matrix(A, At)
                pretty(S, "A + A^T : ")
                # A·A^T
                P = multiply_matrix(A, At)
                pretty(P, "A · A^T:")
            except Exception as e:
                print("연산 실패 : ", e)

        else:
            print("메뉴 번호 다시 확인 바람.")

if __name__ == "__main__":
    menu()

