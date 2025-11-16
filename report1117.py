import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 전역 변수 초기화
N = 0
SET_A = set()

# ----------------------------------------------
# 1. 관계 행렬 입력 및 유연성 기능
# ----------------------------------------------
def setup_set_size():
    global N, SET_A
    while True:
        try:
            size = int(input("분석할 집합 A의 크기 N을 입력하세요. : "))
            if size <= 1:
                print("오류 : 집합의 크기는 2 이상이어야 합니다.")
                continue
            N = size
            SET_A = set(range(1, N + 1))
            print(f"집합 A가 A = {sorted(list(SET_A))} 로 설정되었습니다. (크기 N={N})")
            break
        except ValueError:
            print("오류 : 유효한 정수를 입력해 주세요.")

def input_relation_matrix():
    elements = sorted(list(SET_A))
    print(f"\n{N}x{N} 관계 행렬을 입력합니다. (1 : 관계 있음, 0 : 관계 없음)")
    matrix = []
    for i in range(N):
        while True:
            try:
                row_str = input(f"제 {i+1}행 (원소 {elements[i]}에 대한 관계)을 공백으로 구분하여 입력하세요: ").strip()
                row = [int(x) for x in row_str.split()]
                if len(row) != N or not all(x in [0, 1] for x in row):
                    print(f"오류 : {N}개의 원소(0 또는 1)를 공백으로 구분하여 입력해야 합니다. 다시 입력해 주세요.")
                    continue
                matrix.append(row)
                break
            except ValueError:
                print("오류 : 숫자를 정확히 입력하고 공백으로 구분해 주세요.")
    print("\n입력된 관계 행렬 R :")
    for row in matrix:
        print(row)
    return matrix

# ----------------------------------------------
# 2. 관계의 성질 판별 기능
# ----------------------------------------------

def is_reflexive(R):
    is_ref = all(R[i][i] == 1 for i in range(N))
    print(f"* 반사 관계 (Reflexive): {'O' if is_ref else 'X'}")
    return is_ref

def is_symmetric(R):
    is_sym = all(R[i][j] == R[j][i] for i in range(N) for j in range(N))
    print(f"* 대칭 관계 (Symmetric): {'O' if is_sym else 'X'}")
    return is_sym

def is_transitive(R):
    # Warshall 기반 추이 폐포 계산 함수를 재사용하여 R^+ 와 비교
    R_t = compute_transitive_closure(R)
    # R이 R_t와 같다면 추이 관계 성립
    is_trans = all(R[i][j] == R_t[i][j] for i in range(N) for j in range(N))

    print(f"* 추이 관계 (Transitive) : {'O' if is_trans else 'X'}")
    return is_trans

def is_antisymmetric(R):
    is_antisym = True
    for i in range(N):
        for j in range(N):
            if i != j and R[i][j] == 1 and R[j][i] == 1:
                is_antisym = False
                break
        if not is_antisym:
            break
    print(f"* 반대칭 관계 (Antisymmetric) : {'O' if is_antisym else 'X'}")
    return is_antisym

def check_relation_properties(R, show_summary=True):
    print("\n관계의 성질 판별 결과:")
    is_ref = is_reflexive(R)
    is_sym = is_symmetric(R)
    is_trans = is_transitive(R)
    is_antisym = is_antisymmetric(R)

    is_equiv = is_ref and is_sym and is_trans
    is_partial_order = is_ref and is_antisym and is_trans

    if show_summary:
        print("\n---")
        if is_equiv:
            print("**이 관계는 동치 관계입니다.")
        elif is_partial_order:
            print("**이 관계는 부분 순서 관계입니다.")
        else:
            print("**이 관계는 동치 관계도 부분 순서 관계도 아닙니다.")
        print("---")

    return is_equiv, is_ref, is_sym, is_trans, is_antisym

# ----------------------------------------------
# 3. 동치 관계일 경우 동치류 출력 기능
# ----------------------------------------------
def find_equivalence_classes(R):
    print("\n동치류 (Equivalence Classes) 출력 : ")
    classes = {}
    elements = sorted(list(SET_A))

    for i in range(N):
        element = elements[i]
        eq_class = {elements[j] for j in range(N) if R[i][j] == 1}
        classes[element] = eq_class

    unique_classes = []
    seen_classes = set()

    for eq_class in classes.values():
        frozen_class = frozenset(eq_class)
        if frozen_class not in seen_classes:
            unique_classes.append(eq_class)
            seen_classes.add(frozen_class)

    for element in elements:
        print(f"  - 원소 {element}의 동치류 [{element}]: {classes[element]}")

    print("\n집합 A의 분할 (동치 관계에 의한 Partition) : ")
    for cls in unique_classes:
        print(f"  -> {cls}")

# ----------------------------------------------
# 4. 폐포 구현 기능
# ----------------------------------------------
def print_closure_comparison(R_original, R_closure, closure_type):
    print(f"\n--- {closure_type} 폐포 비교 ---")
    print(f"변환 전 관계 행렬 R : ")
    for row in R_original:
        print(row)
    print(f"변환 후 관계 행렬 {closure_type} 폐포 : ")
    for row in R_closure:
        print(row)
    print("----------------------------------")

def compute_reflexive_closure(R):
    R_closure = [row[:] for row in R]
    for i in range(N):
        R_closure[i][i] = 1
    return R_closure

def compute_symmetric_closure(R):
    R_closure = [row[:] for row in R]
    for i in range(N):
        for j in range(N):
            if R[i][j] == 1 or R[j][i] == 1:
                R_closure[i][j] = 1
    return R_closure

def compute_transitive_closure(R):
    R_closure_np = np.array(R, dtype=int)
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if R_closure_np[i][j] == 0:
                    R_closure_np[i][j] = R_closure_np[i][j] | (R_closure_np[i][k] & R_closure_np[k][j])
    return R_closure_np.tolist()

def compute_equivalence_closure(R):
    R_r = compute_reflexive_closure(R)
    R_rs = compute_symmetric_closure(R_r)
    R_rst = compute_transitive_closure(R_rs)
    return R_rst

def handle_closures(R, is_ref, is_sym, is_trans):
    """
    개별 폐포를 계산한 후,
    - 폐포 행렬 비교 출력
    - 성질 재판별만 수행 (요약 문구 X)
    ※ 동치류 출력은 하지 않는다 (원래 관계가 동치일 때만 main에서 출력)
    """

    # 1. 반사 폐포
    if not is_ref:
        print("\n--- 4-1. 반사 폐포 구현 ---")
        R_ref_closure = compute_reflexive_closure(R)
        print_closure_comparison(R, R_ref_closure, "반사")
        print("반사 폐포로 변환한 후 성질 재판별 : ")
        check_relation_properties(R_ref_closure, show_summary=False)

    # 2. 대칭 폐포
    if not is_sym:
        print("\n--- 4-2. 대칭 폐포 구현 ---")
        R_sym_closure = compute_symmetric_closure(R)
        print_closure_comparison(R, R_sym_closure, "대칭")
        print("대칭 폐포로 변환한 후 성질 재판별 : ")
        check_relation_properties(R_sym_closure, show_summary=False)

    # 3. 추이 폐포
    if not is_trans:
        print("\n--- 4-3. 추이 폐포 구현 ---")
        R_trans_closure = compute_transitive_closure(R)
        print_closure_comparison(R, R_trans_closure, "추이")
        print("추이 폐포로 변환한 후 성질 재판별 : ")
        check_relation_properties(R_trans_closure, show_summary=False)

    # 4. 동치 폐포 (최소 동치 관계) – 여기서도 동치류는 출력하지 않음
    print("\n--- 4-4. 동치 폐포 R^e 구현 ---")
    R_equiv_closure = compute_equivalence_closure(R)
    print_closure_comparison(R, R_equiv_closure, "동치")
    print("동치 폐포 R^e의 성질 재판별 : ")
    check_relation_properties(R_equiv_closure, show_summary=False)

# ----------------------------------------------
# 5. 관계의 시각화 기능
# ----------------------------------------------
def visualize_relation(R, title):
    """visualize relation"""
    elements = sorted(list(SET_A))
    G = nx.DiGraph()

    G.add_nodes_from(elements)

    edges = []
    for i in range(N):
        for j in range(N):
            if R[i][j] == 1:
                u = elements[i]
                v = elements[j]
                edges.append((u, v))
    G.add_edges_from(edges)

    plt.figure(figsize=(7, 7))
    pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='->', arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    plt.title(f"visualize the relation : {title}", fontsize=15)
    plt.axis('off')
    plt.show()

# ----------------------------------------------
# 메인 실행 함수
# ----------------------------------------------
def main():
    try:
        print("==============================================")
        print("관계 분석 및 폐포 구현 프로그램")
        print("==============================================")

        setup_set_size()
        R = input_relation_matrix()

        # 원래 관계 R은 요약까지 출력
        is_equiv, is_ref, is_sym, is_trans, is_antisym = check_relation_properties(R, show_summary=True)

        # 원래 R이 동치일 때만 동치류 출력
        if is_equiv:
            visualize_relation(R, "R")
            find_equivalence_classes(R)
        else:
            handle_closures(R, is_ref, is_sym, is_trans)

        print("\n==============================================")
        print("프로그램이 종료되었습니다.")
        print("==============================================")

    except NameError:
        print("\n\n오류 : 이 프로그램은 행렬 연산을 위해 NumPy와 그래프 시각화를 위해 NetworkX, Matplotlib 라이브러리가 필요합니다.")
        print("터미널/명령 프롬프트에서 'pip install numpy networkx matplotlib' 명령어로 설치 후 다시 실행해 주세요.")
    except Exception as e:
        print(f"\n\n치명적인 오류 발생 : {e}")


if __name__ == "__main__":
    main()