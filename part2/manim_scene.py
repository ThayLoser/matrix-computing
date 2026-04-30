from manim import *
import numpy as np

"""
Manim animation: Matrix Diagonalization  A = P D P^{-1}
Manim Community v0.18+

Run:
    manim -pql matrix_scene.py FullPresentation
    manim -pqh matrix_scene.py FullPresentation
"""


# ── Color palette ─────────────────────────────────────────────────────────
BG      = "#0F1117"
WHITE_S = "#DDE6F0"
BLUE    = "#4EA8DE"   # matrix A
RED     = "#FF6B6B"   # eigenvalue lambda
GREEN   = "#57CC6E"   # eigenvector
ORANGE  = "#F4A14B"   # Gauss pivot / row operation
PURPLE  = "#B48FE0"   # free variable / back-substitution
GOLD    = "#F5C842"   # matrix D
CYAN    = "#56CBF2"   # matrix P
LIME    = "#6BD96B"   # verification / result

config.background_color = BG
config.pixel_height      = 720
config.pixel_width       = 1280
config.frame_height      = 8.0
config.frame_width       = 14.2
config["no_latex_cleanup"] = True


# ════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def _fmt(v):
    """Format float entries cleanly."""
    if isinstance(v, float):
        if abs(v) < 1e-9: return "0"
        return f"{v:.3f}".rstrip('0').rstrip('.')
    return str(v)

def mat_mob(data, color=WHITE_S, sc=0.62, hb=1.15, vb=0.75):
    """Create a Matrix mobject from a 2-D list."""
    m = Matrix([[_fmt(v) for v in row] for row in data],
               h_buff=hb, v_buff=vb)
    m.set_color(color)
    return m.scale(sc)

def mat_lbl(mob, txt, color=WHITE_S, sz=22):
    """Small text label placed above a mobject.
    Tự động dùng MathTex nếu có ký hiệu toán học (^, _, \\, ^{-1}...)"""
    has_math = any(ch in txt for ch in ['^', '_', '\\', '{', '}']) or '^{-1}' in txt or '$' in txt
    if has_math:
        t = MathTex(txt, font_size=sz, color=color)
    else:
        t = Text(txt, font_size=sz, color=color)
    t.next_to(mob, UP, buff=0.18)
    return t

def badge(txt, color, width=4.8):
    """Rounded-rectangle step header, top-left corner."""
    bg = RoundedRectangle(
        width=width, height=0.50, corner_radius=0.13,
        fill_color=color, fill_opacity=0.15,
        stroke_color=color, stroke_width=1.3)
    t = Tex(txt, font_size=20, color=color)
    t.move_to(bg)
    return VGroup(bg, t)

def v_sep(x=0.0, y_top=2.8, y_bot=3.2, opacity=0.22):
    """Vertical dashed separator line."""
    return DashedLine(
        UP * y_top, DOWN * y_bot,
        color=WHITE_S, stroke_opacity=opacity,
        dash_length=0.15, dashed_ratio=0.4
    ).shift(RIGHT * x)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN SCENE
# ════════════════════════════════════════════════════════════════════════════

class DiagonalizationScene(Scene):
    """
    ĐÃ LOẠI BỎ HOÀN TOÀN VIỆC CHUẨN HÓA VECTOR (không còn 1/√2, 1/√5).
    Bây giờ eigenvector dùng trực tiếp kết quả free variable = 1:
        v₁ = [1, 1]   (λ=2)
        v₂ = [2, 1]   (λ=3)
    P = [[1, 2], [1, 1]]  (các cột là eigenvector chưa chuẩn hóa)
    """

    _A         = [[4.0, -2.0], [1.0, 1.0]]
    _L1, _L2   = 2.0, 3.0

    @property
    def _P(self):
        """P giờ dùng vector chưa chuẩn hóa → số nguyên sạch."""
        return [[1.0, 2.0],
                [1.0, 1.0]]

    @property
    def _Pinv(self):
        """Tự động tính inverse từ _P (không cần thay đổi logic)."""
        a, b, c, d = (self._P[0][0], self._P[0][1],
                      self._P[1][0], self._P[1][1])
        dt = a * d - b * c
        return [[d / dt, -b / dt], [-c / dt, a / dt]]

    def construct(self):
        self._s0_title()
        self._s1_intro()
        self._s2_eigenvalues()
        self._s3_gauss()
        self._s4_eigenvectors()
        self._s5_build_PD()
        self._s6_result()

    # S0. TITLE CARD
    def _s0_title(self):
        t1 = Text("Matrix Diagonalization", font_size=52, color=WHITE_S, weight=BOLD)
        t2 = MathTex(r"A = P \, D \, P^{-1}", font_size=44, color=CYAN)
        t2.next_to(t1, DOWN, buff=0.45)
        self.play(Write(t1), run_time=1.0)
        self.play(FadeIn(t2, shift=UP * 0.22), run_time=0.8)
        self.wait(1.0)
        self.play(FadeOut(t1), FadeOut(t2))

    # S1. INTRO
    def _s1_intro(self):
        a_mob = mat_mob(self._A, BLUE, sc=0.90)
        a_lbl = mat_lbl(a_mob, "Matrix  A", BLUE, 22)
        grp_a = VGroup(a_lbl, a_mob).move_to(LEFT * 4.5 + UP * 0.4)
        self.play(FadeIn(grp_a, shift=RIGHT * 0.2), run_time=0.6)

        goal = MathTex(r"A = P \cdot D \cdot P^{-1}", font_size=34, color=CYAN)
        goal.move_to(RIGHT * 1.6 + UP * 0.4)
        arr = Arrow(grp_a.get_right() + RIGHT * 0.12,
                    goal.get_left() - RIGHT * 0.12,
                    color=WHITE_S, stroke_width=2.2,
                    max_tip_length_to_length_ratio=0.18)
        self.play(GrowArrow(arr), run_time=0.4)
        self.play(Write(goal), run_time=0.75)

        comps = [
            (r"P",      "Change-of-basis matrix\n(columns = eigenvectors)",  CYAN,   LEFT*1.0  + DOWN*1.8),
            (r"D",      "Diagonal matrix\n(diagonal = eigenvalues)",         GOLD,   RIGHT*1.6 + DOWN*1.8),
            (r"P^{-1}", "Inverse of  P\n(via Gauss-Jordan)",                 PURPLE, RIGHT*4.2 + DOWN*1.8),
        ]
        for tex, note, col, pos in comps:
            sym  = MathTex(tex, font_size=30, color=col)
            desc = Text(note, font_size=15, color=WHITE_S, line_spacing=1.2)
            desc.next_to(sym, DOWN, buff=0.14)
            g = VGroup(sym, desc).move_to(pos)
            self.play(FadeIn(g, shift=UP * 0.15), run_time=0.38)

        self.wait(0.8)
        self.play(*[FadeOut(m) for m in self.mobjects])

    # S2. EIGENVALUES
    def _s2_eigenvalues(self):
        bdg = badge(r"\textbf{Step 1 — Characteristic Polynomial}", RED, width=5.2)
        bdg.to_corner(UL, buff=0.35)
        self.play(FadeIn(bdg))

        EQ_POS = ORIGIN + UP * 0.5
        eq0 = MathTex(r"\det(A - \lambda I) = 0", font_size=34, color=RED)
        eq0.move_to(EQ_POS)
        self.play(Write(eq0), run_time=0.8)
        self.wait(0.3)

        eq1 = MathTex(r"\det\!\begin{pmatrix}4-\lambda & -2\\[4pt]1 & 1-\lambda\end{pmatrix} = 0",
                      font_size=30, color=WHITE_S)
        eq1.move_to(EQ_POS)
        self.play(ReplacementTransform(eq0, eq1), run_time=1.0)
        self.wait(0.3)

        eq2 = MathTex(r"(4-\lambda)(1-\lambda) + 2 = 0", font_size=32, color=WHITE_S)
        eq2.move_to(EQ_POS)
        self.play(ReplacementTransform(eq1, eq2), run_time=1.0)
        self.wait(0.3)

        eq3 = MathTex(r"\lambda^2 - 5\lambda + 6 = 0", font_size=32, color=WHITE_S)
        eq3.move_to(EQ_POS)
        self.play(ReplacementTransform(eq2, eq3), run_time=1.0)
        self.wait(0.3)

        eq4 = MathTex(r"(\lambda - 2)(\lambda - 3) = 0", font_size=32, color=WHITE_S)
        eq4.move_to(EQ_POS)
        self.play(ReplacementTransform(eq3, eq4), run_time=1.0)
        self.wait(0.4)

        res = MathTex(r"\lambda_1 = 2 \qquad \lambda_2 = 3", font_size=38, color=RED)
        res.move_to(ORIGIN + DOWN * 1.5)
        arr_down = Arrow(eq4.get_bottom() + DOWN * 0.10,
                         res.get_top() - DOWN * 0.05,
                         color=RED, stroke_width=2.2,
                         max_tip_length_to_length_ratio=0.20)
        self.play(GrowArrow(arr_down), run_time=0.4)
        self.play(Write(res), run_time=0.8)

        box_res = SurroundingRectangle(res, color=RED, buff=0.18, stroke_width=1.8)
        self.play(Create(box_res), run_time=0.4)
        self.wait(1.0)
        self.play(*[FadeOut(m) for m in self.mobjects])

    # S3. GAUSS
    def _s3_gauss(self):
        bdg = badge(r"\textbf{Step 2 — Gauss Elimination (REF)}", ORANGE, width=5.0)
        bdg.to_corner(UL, buff=0.35)
        self.play(FadeIn(bdg))

        note = Tex(r"For each eigenvalue, row-reduce $(A - \lambda I)$ to REF",
                   font_size=24, color=WHITE_S)
        note.next_to(bdg, DOWN, buff=0.22).align_to(bdg, LEFT)
        self.play(FadeIn(note), run_time=0.4)

        sep = v_sep(x=-0.5, y_top=2.55, y_bot=3.10)
        self.play(Create(sep), run_time=0.3)

        CX = [-3.6, 3.1]
        Y_TITLE = 2.10; Y_ORIG = 0.90; Y_ARR_T = 0.22; Y_ARR_B = -0.45
        Y_REF = -1.45

        all_mobs = []
        gauss_data = [
            (2.0, [[2., -2.], [1., -1.]], [[2., -2.], [0.,  0.]], r"R_2 \leftarrow R_2 - \tfrac{1}{2}R_1"),
            (3.0, [[1., -2.], [1., -2.]], [[1., -2.], [0.,  0.]], r"R_2 \leftarrow R_2 - R_1"),
        ]

        for i, (lam, M_orig, M_ref, op_tex) in enumerate(gauss_data):
            cx = CX[i]
            t_lam = MathTex(rf"\lambda_{i+1} = {int(lam)}", font_size=26, color=RED)
            t_lam.move_to([cx, Y_TITLE, 0])
            self.play(Write(t_lam), run_time=0.40)

            m_orig = mat_mob(M_orig, BLUE, sc=0.66)
            m_orig.move_to([cx, Y_ORIG, 0])
            lb_orig = mat_lbl(m_orig, f"A - {int(lam)}I", BLUE, 19)
            self.play(FadeIn(VGroup(lb_orig, m_orig)), run_time=0.45)

            arr = Arrow([cx, Y_ARR_T, 0], [cx, Y_ARR_B, 0],
                        color=ORANGE, stroke_width=2.4,
                        max_tip_length_to_length_ratio=0.24)
            self.play(GrowArrow(arr), run_time=0.38)

            op_mob = MathTex(op_tex, font_size=17, color=ORANGE)
            op_mob.next_to(arr, RIGHT, buff=0.10).set_max_width(2.6)
            self.play(FadeIn(op_mob), run_time=0.32)

            m_ref = mat_mob(M_ref, ORANGE, sc=0.66)
            m_ref.move_to([cx, Y_REF, 0])
            lb_ref = mat_lbl(m_ref, "REF", ORANGE, 19)

            m_orig_copy = m_orig.copy().set_color(ORANGE)
            self.play(ReplacementTransform(m_orig_copy, m_ref), FadeIn(lb_ref), run_time=1.0)

            ents = m_ref.get_entries()
            pr = SurroundingRectangle(ents[0], color=ORANGE, buff=0.08, stroke_width=1.6)
            fr = SurroundingRectangle(ents[1], color=PURPLE, buff=0.08, stroke_width=1.6)
            pt = Text("pivot", font_size=10, color=ORANGE)
            ft = Text("free",  font_size=10, color=PURPLE)
            pt.next_to(pr, DOWN, buff=0.07)
            ft.next_to(fr, DOWN, buff=0.07)
            pt.set_x(pr.get_center()[0])
            ft.set_x(fr.get_center()[0])

            self.play(Create(pr), Create(fr), run_time=0.35)
            self.play(Write(pt), Write(ft), run_time=0.30)

            all_mobs += [t_lam, lb_orig, m_orig, arr, op_mob, lb_ref, m_ref, pr, fr, pt, ft]

        self.wait(0.8)
        self.play(FadeOut(*all_mobs, bdg, note, sep), run_time=0.5)

    # S4. EIGENVECTORS – ĐÃ LOẠI BỎ CHUẨN HÓA
    def _s4_eigenvectors(self):
        bdg = badge(r"\textbf{Step 3 — Back Substitution } $\rightarrow$ \textbf{ Eigenvectors}", GREEN, width=6.0)
        bdg.to_corner(UL, buff=0.35)
        self.play(FadeIn(bdg))

        idea = Text("Set free variable = 1  ->  solve for pivot variable",
                    font_size=19, color=WHITE_S)
        idea.next_to(bdg, DOWN, buff=0.22).align_to(bdg, LEFT)
        self.play(FadeIn(idea), run_time=0.40)

        sep = v_sep(x=-0.5, y_top=2.55, y_bot=3.10)
        self.play(Create(sep), run_time=0.25)

        Y_TITLE = 2.10; Y_REF = 1.22; Y_SUB = 0.15; Y_VEC = -1.10; Y_CHK = -2.20
        CX = [-3.6, 3.1]
        all_mobs = []

        eig_data = [
            (1, 2, r"2x_1 - 2x_2 = 0", r"x_2 = 1 \;\Rightarrow\; x_1 = 1",
             r"\mathbf{v}_1 = \begin{pmatrix}1\\[3pt]1\end{pmatrix}",          
             r"A\mathbf{v}_1 = 2\,\mathbf{v}_1 \;\checkmark"),
            (2, 3, r"x_1 - 2x_2 = 0", r"x_2 = 1 \;\Rightarrow\; x_1 = 2",
             r"\mathbf{v}_2 = \begin{pmatrix}2\\[3pt]1\end{pmatrix}",
             r"A\mathbf{v}_2 = 3\,\mathbf{v}_2 \;\checkmark"),
        ]

        for i, (idx, lam, ref_tex, sub_tex, vec_tex, chk_tex) in enumerate(eig_data):
            cx = CX[i]
            t_lam = MathTex(rf"\lambda_{idx} = {lam}", font_size=26, color=RED)
            t_lam.move_to([cx, Y_TITLE, 0])
            self.play(Write(t_lam), run_time=0.38)

            eq_ref = MathTex(r"\text{REF: }" + ref_tex, font_size=22, color=WHITE_S)
            eq_ref.move_to([cx, Y_REF, 0]).set_max_width(3.5)
            self.play(Write(eq_ref), run_time=0.50)

            eq_sub = MathTex(sub_tex, font_size=22, color=PURPLE)
            eq_sub.move_to([cx, Y_SUB, 0]).set_max_width(3.5)
            self.play(ReplacementTransform(eq_ref.copy(), eq_sub), run_time=0.90)

            vec = MathTex(vec_tex, font_size=27, color=GREEN)
            vec.move_to([cx, Y_VEC, 0]).set_max_width(3.2)
            box_v = SurroundingRectangle(vec, color=GREEN, buff=0.14, stroke_width=1.6)
            self.play(ReplacementTransform(eq_sub.copy(), vec), run_time=0.90)
            self.play(Create(box_v), run_time=0.30)

            chk = MathTex(chk_tex, font_size=20, color=LIME)
            chk.move_to([cx, Y_CHK, 0])
            self.play(FadeIn(chk, shift=UP * 0.12), run_time=0.32)

            all_mobs += [t_lam, eq_ref, eq_sub, vec, box_v, chk]

        self.wait(0.9)
        self.play(FadeOut(*all_mobs, bdg, idea, sep), run_time=0.5)

    # S5. BUILD P, D, P^{-1} 
    def _s5_build_PD(self):
        bdg = badge(r"\textbf{Step 4 — Build $P$, $D$, and $P^{-1}$}", CYAN, width=5.2)
        bdg.to_corner(UL, buff=0.35)
        self.play(FadeIn(bdg))

        Pi = self._Pinv
        Pdata  = [[1.0, 2.0], [1.0, 1.0]]
        Ddata  = [[2.0, 0.0], [0.0, 3.0]]
        PIdata = [[round(Pi[0][0],3), round(Pi[0][1],3)],
                  [round(Pi[1][0],3), round(Pi[1][1],3)]]

        v1_tex = MathTex(r"\mathbf{v}_1 = \begin{pmatrix}1\\1\end{pmatrix}", font_size=26, color=GREEN)
        v2_tex = MathTex(r"\mathbf{v}_2 = \begin{pmatrix}2\\1\end{pmatrix}", font_size=26, color=GREEN)
        v1_tex.move_to(LEFT * 4.2 + UP * 1.55)
        v2_tex.move_to(LEFT * 1.4 + UP * 1.55)
        self.play(FadeIn(v1_tex, v2_tex), run_time=0.5)

        p_mob = mat_mob(Pdata, CYAN, sc=0.76, hb=1.65)
        p_mob.move_to(LEFT * 3.9 + DOWN * 0.45)
        p_lbl = mat_lbl(p_mob, "P", CYAN, 26)
        p_note = Text("(each column = one eigenvector)", font_size=15, color=WHITE_S)
        p_note.next_to(p_mob, DOWN, buff=0.20)

        self.play(ReplacementTransform(VGroup(v1_tex, v2_tex).copy(), p_mob),
                  FadeIn(p_lbl), run_time=1.10)
        self.play(FadeOut(v1_tex, v2_tex), FadeIn(p_note), run_time=0.38)

        pe = p_mob.get_entries()
        rc1 = SurroundingRectangle(VGroup(pe[0], pe[2]), color=RED, buff=0.08)
        rc2 = SurroundingRectangle(VGroup(pe[1], pe[3]), color=LIME, buff=0.08)
        tg1 = MathTex(r"\lambda_1=2", font_size=17, color=RED)
        tg2 = MathTex(r"\lambda_2=3", font_size=17, color=LIME)
        tg1.next_to(rc1, DOWN, buff=0.06)
        tg2.next_to(rc2, DOWN, buff=0.06)
        self.play(Create(rc1), Create(rc2), run_time=0.35)
        self.play(Write(tg1), Write(tg2), run_time=0.30)

        l1_tex = MathTex(r"\lambda_1=2", font_size=25, color=RED)
        l2_tex = MathTex(r"\lambda_2=3", font_size=25, color=RED)
        l1_tex.move_to(RIGHT * 1.2 + UP * 1.55)
        l2_tex.move_to(RIGHT * 3.4 + UP * 1.55)
        self.play(FadeIn(l1_tex, l2_tex), run_time=0.38)

        d_mob = mat_mob(Ddata, GOLD, sc=0.76)
        d_mob.move_to(RIGHT * 1.6 + DOWN * 0.45)
        d_lbl = mat_lbl(d_mob, "D", GOLD, 26)
        d_note = Text("(eigenvalues on the diagonal)", font_size=15, color=WHITE_S)
        d_note.next_to(d_mob, DOWN, buff=0.20)

        self.play(ReplacementTransform(VGroup(l1_tex, l2_tex).copy(), d_mob),
                  FadeIn(d_lbl), run_time=1.10)
        self.play(FadeOut(l1_tex, l2_tex), FadeIn(d_note), run_time=0.38)

        de = d_mob.get_entries()
        rd1 = SurroundingRectangle(de[0], color=GOLD, buff=0.09)
        rd2 = SurroundingRectangle(de[3], color=GOLD, buff=0.09)
        self.play(Create(rd1), Create(rd2), run_time=0.35)

        pi_mob = mat_mob(PIdata, PURPLE, sc=0.76)
        pi_mob.move_to(RIGHT * 5.0 + DOWN * 0.45)
        pi_lbl = mat_lbl(pi_mob, "P^{-1}", PURPLE, 24)
        pi_note = Text("Gauss-Jordan on  [P | I]", font_size=15, color=WHITE_S)
        pi_note.next_to(pi_mob, DOWN, buff=0.20)

        p_copy = p_lbl.copy().set_color(PURPLE)
        self.play(ReplacementTransform(p_copy, pi_mob),
                  FadeIn(pi_lbl, pi_note), run_time=1.00)

        self.wait(1.0)
        self.play(*[FadeOut(m) for m in self.mobjects])

    # S6. RESULT
    def _s6_result(self):
        bdg = badge(r"\textbf{Result — $A = P D P^{-1}$}", LIME, width=4.4)
        bdg.to_corner(UL, buff=0.35)
        self.play(FadeIn(bdg))

        formula = MathTex(r"A \;=\; P \cdot D \cdot P^{-1}", font_size=36, color=WHITE_S)
        formula.move_to(UP * 2.20)
        self.play(Write(formula), run_time=0.75)

        Pi = self._Pinv
        Pdata  = [[1.0, 2.0], [1.0, 1.0]]
        Ddata  = [[2.0, 0.0], [0.0, 3.0]]
        PIdata = [[round(Pi[0][0],3), round(Pi[0][1],3)],
                  [round(Pi[1][0],3), round(Pi[1][1],3)]]

        SC = 0.60
        m_a  = mat_mob(self._A,  BLUE,   SC)
        m_p  = mat_mob(Pdata,    CYAN,   SC)
        m_d  = mat_mob(Ddata,    GOLD,   SC)
        m_pi = mat_mob(PIdata,   PURPLE, SC)

        eq_s = MathTex(r"=", font_size=30, color=LIME)
        dot1 = MathTex(r"\cdot", font_size=30, color=WHITE_S)
        dot2 = MathTex(r"\cdot", font_size=30, color=WHITE_S)

        row = VGroup(m_a, eq_s, m_p, dot1, m_d, dot2, m_pi)
        row.arrange(RIGHT, buff=0.28).move_to(DOWN * 0.25)

        lb_a  = mat_lbl(m_a,  "A",       BLUE,   17)
        lb_p  = mat_lbl(m_p,  "P",       CYAN,   17)
        lb_d  = mat_lbl(m_d,  "D",       GOLD,   17)
        lb_pi = mat_lbl(m_pi, "P^{-1}",  PURPLE, 17)

        for m, lb in [(m_a, lb_a), (m_p, lb_p), (m_d, lb_d), (m_pi, lb_pi)]:
            self.play(FadeIn(VGroup(lb, m), shift=UP * 0.15), run_time=0.42)
        for s in [eq_s, dot1, dot2]:
            self.play(FadeIn(s), run_time=0.20)

        verify = Tex(r"\textbf{Verified: } $P D P^{-1} = A$ \textbf{ (error $< 10^{-9}$)}",
                     font_size=24, color=LIME)
        verify.to_edge(DOWN, buff=0.55)
        self.play(Write(verify), run_time=0.55)

        self.wait(1.8)
        self.play(*[FadeOut(m) for m in self.mobjects])

class LUDecomposition(Scene):
    def construct(self):
        # Cấu hình chung, tiêu đề
        SCALE_FACTOR = 0.7

        title = Text("Phân rã Ma trận PA = LU", font="Arial", color=BLUE).scale(0.9).to_edge(UP)
        sub_title = Text("(Kỹ thuật chọn phần tử trội - Partial Pivoting)", font="Arial", font_size=20).next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(FadeIn(sub_title))
        self.wait(2)
        
        # Khởi tạo các ma trận P, L, U và A
        # Ma trận A
        A_val = [[0, 2, 1], [4, -6, 0], [-2, 7, 2]]
        A_mat = Matrix(A_val).scale(SCALE_FACTOR)
        A_label = MathTex("A =").next_to(A_mat, LEFT).scale(SCALE_FACTOR)
        A_group = VGroup(A_label, A_mat).center()
        
        self.play(Write(A_group))
        self.wait(2)
        
        self.play(A_group.animate.move_to(LEFT * 3.5 + UP * 1))
        
        # Khởi tạo P, L, U
        p_mat = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).scale(SCALE_FACTOR)
        l_mat = Matrix([[1, 0, 0], ["l_{21}", 1, 0], ["l_{31}", "l_{32}", 1]]).scale(SCALE_FACTOR)
        u_mat = Matrix(A_val).scale(SCALE_FACTOR)
        
        p_label = MathTex("P=").next_to(p_mat, LEFT).scale(SCALE_FACTOR)
        l_label = MathTex("L=").next_to(l_mat, LEFT).scale(SCALE_FACTOR)
        u_label = MathTex("U=").next_to(u_mat, LEFT).scale(SCALE_FACTOR)
        
        p_group = VGroup(p_label, p_mat)
        l_group = VGroup(l_label, l_mat)
        u_group = VGroup(u_label, u_mat)
        
        right_side = VGroup(u_group, l_group, p_group).arrange(DOWN, buff=0.3).move_to(RIGHT * 2 + UP * 0.5)
        
        self.play(FadeOut(title), FadeOut(sub_title))
        msg = Text("Khởi tạo: U = A, L = I, P = I", font="Arial", font_size=24, color=YELLOW).to_edge(DOWN)
        self.play(Write(msg))
        self.play(FadeIn(right_side))
        self.wait(3)
        self.play(FadeOut(msg))

        # Khử cột 1
        step1_msg = Text("Bước 1: Khử cột 1", font="Arial", font_size=24, color=GREEN).to_edge(DOWN)
        self.play(Write(step1_msg))
        
        pivot_rect = SurroundingRectangle(u_mat.get_entries()[0], color=RED)
        target_rect = SurroundingRectangle(u_mat.get_entries()[3], color=YELLOW)
        self.play(Create(pivot_rect), Create(target_rect))
        
        swap_info = Text("Đổi Hàng 1 <-> Hàng 2 (Vì |4| là lớn nhất)", font="Arial", font_size=20).next_to(step1_msg, UP)
        self.play(Write(swap_info))
        self.wait(3)
        
        # Hoán vị (Visual)
        self.play(
            u_mat.get_rows()[0].animate.move_to(u_mat.get_rows()[1]),
            u_mat.get_rows()[1].animate.move_to(u_mat.get_rows()[0]),
            p_mat.get_rows()[0].animate.move_to(p_mat.get_rows()[1]),
            p_mat.get_rows()[1].animate.move_to(p_mat.get_rows()[0]),
            run_time=2
        )
        
        u_val_step1 = [[4, -6, 0], [0, 2, 1], [-2, 7, 2]]
        u_mat_new = Matrix(u_val_step1).scale(SCALE_FACTOR).move_to(u_mat)
        p_val_step1 = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
        p_mat_new = Matrix(p_val_step1).scale(SCALE_FACTOR).move_to(p_mat)
        
        # Dọn dẹp khung chữ và đồng bộ (Logic)
        self.play(FadeOut(pivot_rect), FadeOut(target_rect), FadeOut(swap_info), run_time=0.5)
        u_mat.become(u_mat_new)
        p_mat.become(p_mat_new)
        
        # Trình bày công thức khử Bước 1
        calc_l = MathTex("l_{21} = 0/4 = 0, \\quad l_{31} = -2/4 = -0.5").scale(0.8)
        op1 = MathTex("R_2 \\leftarrow R_2 - l_{21}R_1 \\implies R_2 = [0, 2, 1]").scale(0.8)
        op2 = MathTex("R_3 \\leftarrow R_3 - l_{31}R_1 \\implies R_3 = [0, 4, 2]").scale(0.8).set_color(ORANGE)
        
        step1_calculations = VGroup(calc_l, op1, op2).arrange(DOWN, buff=0.3).next_to(A_group, DOWN, buff=1)
        
        self.play(Write(step1_calculations[0]))
        self.wait(2)
        
        l21 = MathTex("0").scale(SCALE_FACTOR).move_to(l_mat.get_entries()[3])
        l31 = MathTex("-0.5").scale(0.5).move_to(l_mat.get_entries()[6])
        self.play(Transform(l_mat.get_entries()[3], l21), Transform(l_mat.get_entries()[6], l31))
        
        self.play(Write(step1_calculations[1]))
        self.wait(2)
        self.play(Write(step1_calculations[2]))
        self.wait(3)
        
        # Cập nhật U
        u3_new = [MathTex("0"), MathTex("4"), MathTex("2")]
        for i in range(3):
            u3_new[i].scale(SCALE_FACTOR).move_to(u_mat.get_entries()[6+i]).set_color(ORANGE)
            self.play(Transform(u_mat.get_entries()[6+i], u3_new[i]), run_time=0.4)
        
        self.wait(3)
        self.play(FadeOut(step1_calculations), FadeOut(step1_msg))

        # Khử cột 2
        step2_msg = Text("Bước 2: Khử cột 2", font="Arial", font_size=24, color=GREEN).to_edge(DOWN)
        self.play(Write(step2_msg))
        
        pivot_rect = SurroundingRectangle(u_mat.get_entries()[4], color=RED)
        target_rect = SurroundingRectangle(u_mat.get_entries()[7], color=YELLOW)
        self.play(Create(pivot_rect), Create(target_rect))
        
        swap_info2 = Text("Đổi Hàng 2 <-> Hàng 3 (Vì |4| > |2|)", font="Arial", font_size=20).next_to(step2_msg, UP)
        self.play(Write(swap_info2))
        self.wait(3)
        
        # Hoán vị (Visual)
        self.play(
            u_mat.get_rows()[1].animate.move_to(u_mat.get_rows()[2]),
            u_mat.get_rows()[2].animate.move_to(u_mat.get_rows()[1]),
            p_mat.get_rows()[1].animate.move_to(p_mat.get_rows()[2]),
            p_mat.get_rows()[2].animate.move_to(p_mat.get_rows()[1]),
            l_mat.get_entries()[3].animate.move_to(l_mat.get_entries()[6]),
            l_mat.get_entries()[6].animate.move_to(l_mat.get_entries()[3]),
            run_time=2
        )
        
        u_val_step2 = [[4, -6, 0], [0, 4, 2], [0, 2, 1]]
        u_mat_step2 = Matrix(u_val_step2).scale(SCALE_FACTOR).move_to(u_mat)
        p_val_step2 = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
        p_mat_step2 = Matrix(p_val_step2).scale(SCALE_FACTOR).move_to(p_mat)
        
        # Dọn dẹp khung chữ và đồng bộ (Logic)
        self.play(FadeOut(pivot_rect), FadeOut(target_rect), FadeOut(swap_info2), run_time=0.5)
        u_mat.become(u_mat_step2)
        p_mat.become(p_mat_step2)
        
        self.wait(1)
        
        # Trình bày công thức khử Bước 2
        calc_l32 = MathTex("l_{32} = 2 / 4 = 0.5").scale(0.8)
        op3 = MathTex("R_3 \\leftarrow R_3 - l_{32}R_2 \\implies R_3 = [0, 0, 0]").scale(0.8).set_color(RED)

        step2_calculations = VGroup(calc_l32, op3).arrange(DOWN, buff=0.3).next_to(A_group, DOWN, buff=1)
        
        self.play(Write(step2_calculations[0]))
        self.wait(2)
        
        l32 = MathTex("0.5").scale(0.5).move_to(l_mat.get_entries()[7])
        self.play(Transform(l_mat.get_entries()[7], l32))
        
        self.play(Write(step2_calculations[1]))
        self.wait(3)
        
        # Cập nhật U
        for i in range(3):
            val = MathTex("0").scale(SCALE_FACTOR).move_to(u_mat.get_entries()[6+i]).set_color(RED)
            self.play(Transform(u_mat.get_entries()[6+i], val), run_time=0.4)
        
        self.wait(3)
        self.play(FadeOut(step2_calculations), FadeOut(step2_msg))

        # Trình bày kết quả cuối cùng
        # Dọn dẹp toàn bộ màn hình
        self.play(
            FadeOut(A_group), FadeOut(u_group), FadeOut(l_group), FadeOut(p_group)
        )
        self.wait(1)

        # PA =
        pa_label = MathTex("P \\cdot A =").set_color(YELLOW)
        p_mat_f = Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        a_mat_f = Matrix([[0, 2, 1], [4, -6, 0], [-2, 7, 2]])
        eq1 = MathTex("=")
        res_mat = Matrix([[4, -6, 0], [-2, 7, 2], [0, 2, 1]])
        pa_eq = VGroup(pa_label, p_mat_f, a_mat_f, eq1, res_mat).arrange(RIGHT, buff=0.2).scale(0.75)

        # LU = 
        lu_label = MathTex("L \\cdot U =").set_color(YELLOW)
        l_mat_f = Matrix([[1, 0, 0], ["-0.5", 1, 0], [0, "0.5", 1]])
        u_mat_f = Matrix([[4, -6, 0], [0, 4, 2], [0, 0, 0]])
        
        # Tô đỏ số 0 của U để nhấn mạnh
        u_mat_f.get_entries()[3].set_color(RED)
        u_mat_f.get_entries()[6].set_color(RED)
        u_mat_f.get_entries()[7].set_color(RED)
        eq2 = MathTex("=")
        res_mat2 = Matrix([[4, -6, 0], [-2, 7, 2], [0, 2, 1]])
        lu_eq = VGroup(lu_label, l_mat_f, u_mat_f, eq2, res_mat2).arrange(RIGHT, buff=0.2).scale(0.75)
        final_equations = VGroup(pa_eq, lu_eq).arrange(DOWN, buff=1).center().shift(UP*0.5)

        self.play(Write(pa_label))
        self.play(FadeIn(p_mat_f), FadeIn(a_mat_f))
        self.wait(1)
        self.play(Write(eq1), FadeIn(res_mat))
        self.wait(2)

        self.play(Write(lu_label))
        self.play(FadeIn(l_mat_f), FadeIn(u_mat_f))
        self.wait(1)
        self.play(Write(eq2), FadeIn(res_mat2))
        self.wait(2)

        # kết luận
        final_txt = Text("=> Kết luận: PA = LU", font="Arial", color=GREEN).scale(1.2).next_to(final_equations, DOWN, buff=1)
        
        box1 = SurroundingRectangle(res_mat, color=BLUE, buff=0.1)
        box2 = SurroundingRectangle(res_mat2, color=BLUE, buff=0.1)
        
        self.play(Create(box1), Create(box2))
        self.play(Write(final_txt))
        
        self.wait(5)


# Cấu hình font
VFONT = "Arial"

# ─── BẢNG MÀU (PALETTE) ──────────────────────────────────────────────────────
BG       = "#0d1117"
GOLD     = "#f0c060"
CYAN     = "#56cfe1"
MINT     = "#72efdd"
PINK     = "#ff6b9d"
LAVENDER = "#c77dff"
WHITE_C  = "#e8eaf6"
DIM      = "#8892a4"

class SVD(Scene):
    def construct(self):
        self.camera.background_color = BG # type: ignore
        # Part 1: QR Decomposition
        A = np.array([[1, 1, 2], [2, -1, 1], [-2, 4, 1]], dtype=float)
        # Tiêu đề QR
        title_qr = Text("Phân rã QR: Trực giao hóa Gram-Schmidt", font=VFONT, font_size=38, color=GOLD, weight=BOLD)
        # Mục tiêu
        sub_qr = Text("Mục tiêu: Tìm Q, R sao cho A = Q x R", font=VFONT, font_size=26, color=CYAN).next_to(title_qr, DOWN)
        # Giải thích Q,R
        sub_qr2 = Text("Trong đó: Q: Ma trận trực giao, R: Ma trận tam giác trên", font=VFONT, font_size=22, color=MINT).next_to(sub_qr, DOWN, buff=0.5)
        
        # Giới thiệu QR
        self.play(Write(title_qr), run_time=1)
        self.play(Write(sub_qr), run_time=1)
        self.play(Write(sub_qr2), run_time=1)
        self.wait(1.5); self.play(FadeOut(title_qr), FadeOut(sub_qr), FadeOut(sub_qr2))

        # Gioi thiệu ma trận A 

        A_mob = Matrix(A, left_bracket="(",right_bracket=")").scale(0.8).set_color(WHITE_C)
        A_label = Text("A", font=VFONT, font_size=24, color=WHITE_C, weight=BOLD).next_to(A_mob, UP, buff=0.3)
        A_group = VGroup(A_mob, A_label)
        self._show_step("Ma trận A đầu vào")
        self.play(Create(A_group), run_time=1)
        self.wait(0.5); self._clear_step()
        # Step 1: Tính Q
        # Animation đưa A sang trái
        self._show_step("Bước 1: Tính ma trận Q")
        self.play(A_group.animate.to_edge(LEFT, buff=1.5).to_edge(UP*1.8, buff=0.5), run_time=1)
        self.wait(0.5)
        # Step 2: Chuyển từng cột của A thành vector u 
        columns_data = [A[:, i] for i in range(A.shape[1])]
        u_groups = []
        rect = SurroundingRectangle(A_mob.get_columns()[0], color=RED, buff=0.2)
        rect.set_opacity(0)
        for i, col_data in enumerate(columns_data):
            u_mat = Matrix([col_data], left_bracket="(", right_bracket=")").scale(0.8).set_color(CYAN)
            u_label = MathTex(f"u_{i+1} = ", color=CYAN)
            u_label.next_to(u_mat, LEFT, buff=0.15)
            curr_u_group = VGroup(u_mat, u_label)

            if i == 0:
                curr_u_group.next_to(A_group, RIGHT, buff=2.5).shift(UP*0.8)
            else:
                curr_u_group.next_to(u_groups[i-1], DOWN, buff=0.15).align_to(u_groups[i-1], LEFT)
            
            u_mat.set_width(u_groups[0][0].get_width() if i > 0 else u_mat.get_width())

            rect.replace(A_mob.get_columns()[i], stretch=True).set_stroke(color=RED)
            self.play(rect.animate.set_stroke(opacity=1), run_time=0.5)
            
            self.play(rect.animate.replace(curr_u_group, stretch=True).set_color(CYAN),run_time=0.5)
            self.play(Create(curr_u_group), rect.animate.set_stroke(opacity=0), run_time=0.5)

            u_groups.append(curr_u_group)

        self.wait(1)

        # Step 3: v_i = u_i - sum(<u_i, v_j> / ||v_j||^2 * v_j) với j < i

        # Hiển thị công thức
        ortho_text = Text("Trực giao hóa Gram-Schmidt", font=VFONT, font_size=28, color=GREEN)
        ortho_formula = MathTex(r"v_i = u_i - \sum_{j < i} \frac{\langle u_i, v_j \rangle}{||v_j||^2} v_j", font_size=28, color=GREEN).next_to(ortho_text, DOWN, buff=0.15)
        ortho_group = VGroup(ortho_text, ortho_formula).next_to(A_group, DOWN, buff=0.5)
        self.play(Create(ortho_group), run_time=1)


        u_vectors_data = [col_data for col_data in columns_data] # Dữ liệu vector u để tính v
        v_list_raw = []   # Lưu dữ liệu vector v thô để tính cho v tiếp theo
        v_groups = []   # Lưu Mobject để hiển thị

        for i in range(len(u_vectors_data)):
            ui_raw = u_vectors_data[i]
            vi_raw = ui_raw.copy()
            
            for vj_raw in v_list_raw:
                proj = (np.dot(ui_raw.T, vj_raw) / np.dot(vj_raw.T, vj_raw)) * vj_raw
                vi_raw = vi_raw - proj
            
            v_list_raw.append(vi_raw)

            vi_rounded = np.round(vi_raw.flatten(), 2) 
            
            v_mat = Matrix([vi_rounded], left_bracket="(", right_bracket=")").scale(0.8).set_color(GREEN)
            
            v_label = MathTex(f"v_{i+1} =", color=GREEN)
            curr_v_group = VGroup(v_label, v_mat).arrange(RIGHT, buff=0.2)

            if i == 0:
                curr_v_group.next_to(ortho_group, DOWN, buff=0.25)
            else:
                curr_v_group.next_to(v_groups[i-1], DOWN, buff=0.15).align_to(v_groups[i-1], LEFT)

            rect = SurroundingRectangle(u_groups[i], color=RED, buff=0.1)
            self.play(Create(rect), run_time=0.5)
            self.wait(0.2)
            self.play(
                FadeOut(rect),
                Create(curr_v_group),
                run_time=0.5
            )
            v_groups.append(curr_v_group)

        # Chuẩn hóa v_i để được q_i
        # Hiển thị công thức
        norm_text = Text("Chuẩn hóa", font=VFONT, font_size=28, color=ORANGE)
        norm_formula = MathTex(r"q_i = \frac{{v_i}}{{||v_i||}}", font_size=28, color=ORANGE).next_to(norm_text, DOWN, buff=0.15)
        norm_group = VGroup(norm_text, norm_formula).next_to(u_groups[len(u_groups)-1], DOWN, buff=0.5)
        self.play(Create(norm_group), run_time=1)

        q_groups = []
        q_list_raw = []

        for i in range(len(v_list_raw)):
            vi_raw = v_list_raw[i]
            norm_vi = np.linalg.norm(vi_raw)
            qi_raw = vi_raw / norm_vi
            q_list_raw.append(qi_raw)
            qi_rounded = np.round(qi_raw.flatten(), 2)
            
            q_mat = Matrix([qi_rounded], left_bracket="(", right_bracket=")").scale(0.8).set_color(ORANGE)
            q_label = MathTex(fr"q_{i+1} =", color=ORANGE).scale(0.8)
            curr_q_group = VGroup(q_label, q_mat).arrange(RIGHT, buff=0.2)

            if i == 0:
                curr_q_group.next_to(norm_group, DOWN, buff=0.25)
            else:
                curr_q_group.next_to(q_groups[i-1], DOWN, buff=0.15).align_to(q_groups[i-1], LEFT)

            rect = SurroundingRectangle(v_groups[i], color=RED, buff=0.1)
            
            self.play(Create(rect), run_time=0.5)
            self.wait(0.2)
            
            self.play(
                FadeOut(rect),
                Create(curr_q_group),
                run_time=0.5
            )
            
            q_groups.append(curr_q_group)


        self.wait(1.25)
        self.play(
            FadeOut(A_group), 
            *[FadeOut(g) for g in u_groups], 
            *[FadeOut(g) for g in v_groups], 
            FadeOut(ortho_group), 
            FadeOut(norm_group),
            run_time=1
        )


        # --- TIẾP TỤC SAU PHẦN FadeOut ---
        
        # 1. Đưa các q_i về vị trí trung tâm để chuẩn bị ghép thành ma trận
        q_final_group = VGroup(*q_groups)
        self.play(
            q_final_group.animate.arrange(DOWN, buff=0.5).center(),
            run_time=1
        )
        # Ẩn các q_i


        self.wait(0.5)

        # 2. Tạo ma trận Q từ dữ liệu q_list_raw
        # q_list_raw chứa các mảng ngang, ta cần chuyển chúng thành các cột của Q
        Q_data = np.column_stack(q_list_raw)
        
        Q_mob = Matrix(
            np.round(Q_data, 2), 
            left_bracket="(", 
            right_bracket=")"
        ).scale(0.8).set_color(ORANGE)
        
        Q_label = MathTex("Q =", color=ORANGE).next_to(Q_mob, LEFT, buff=0.3)
        full_Q_group = VGroup(Q_label, Q_mob).center()
        
        self.play(
            ReplacementTransform(q_final_group[0], Q_mob.get_columns()[0]),
            run_time=1
        )
        self.play(
            ReplacementTransform(q_final_group[1], Q_mob.get_columns()[1]),
            run_time=1
        )
        self.play(
            ReplacementTransform(q_final_group[2], Q_mob.get_columns()[2]),
            run_time=1
        )
        

        self.play(
            Write(Q_label),
            Create(Q_mob.get_brackets()),
            run_time=0.5
        )
        self.wait(1)

        
        self.play(FadeOut(full_Q_group))
        self._clear_step()

        self.wait(2)


        # Tính R

        # B1: Lập R nxn
        self._show_step("Bước 2: Tính ma trận R")

        n = A.shape[1]
        R_data = np.zeros((n, n), dtype=float)

        # Hiển thị công thức R_ij = <q_i, u_j> (i <= j)
        r_text = Text("Tính R từ Q và các cột của A", font=VFONT, font_size=28, color=LAVENDER)
        r_formula = MathTex(r"R_{ij} = \langle q_i, u_j \rangle \quad (i \le j), \quad R_{ij}=0 \ (i>j)",font_size=28,color=LAVENDER).next_to(r_text, DOWN, buff=0.15)
        r_group = VGroup(r_text, r_formula).to_edge(UP, buff=1.35)
        self.play(Create(r_group), run_time=1.2)

        R_mob = Matrix(np.round(R_data, 2), left_bracket="(", right_bracket=")").scale(0.8).set_color(PINK)
        R_label = MathTex("R =", color=PINK).next_to(R_mob, LEFT, buff=0.3)
        R_group_mob = VGroup(R_label, R_mob).next_to(r_group, DOWN, buff=0.6)
        self.play(Write(R_label), Create(R_mob), run_time=1)

        # B2: với i <= j, R[i][j] = <u[j], q[i]>
        # Dựng lại u_data và q_data để tính
        u_data = [A[:, j].reshape(-1, 1) for j in range(n)]      # u_j là cột j của A
        q_data = [q_list_raw[i].reshape(-1, 1) for i in range(n)]  # q_i là cột i của Q

        cell_rect = SurroundingRectangle(R_mob.get_entries()[0], color=RED, buff=0.2)
        cell_rect.set_opacity(0)
        expr = MathTex("w", color=CYAN).scale(0.7).set_opacity(0).next_to(R_group_mob, DOWN, buff=0.25).align_to(R_group_mob, LEFT)
        for i in range(n):
            for j in range(i, n):
                val = float(np.dot(q_data[i].T, u_data[j]).item())
                R_data[i, j] = val

                # Tạo R mới để transform
                R_new = Matrix(np.round(R_data, 2), left_bracket="(", right_bracket=")").scale(0.8).set_color(PINK)
                R_new.move_to(R_mob)

                cell_rect = SurroundingRectangle(R_mob.get_entries()[i * n + j], color=RED, buff=0.2)

                new_expr = MathTex(fr"R_{{{i+1},{j+1}}} = \langle q_{i+1}, u_{j+1} \rangle = {np.round(val, 2)}", color=CYAN).scale(0.7)
                new_expr.next_to(R_group_mob, DOWN, buff=0.25).align_to(R_group_mob, LEFT)
                

                self.play(Create(cell_rect), run_time=0.25)
                self.play(Transform(expr, new_expr), run_time=0.35)
                self.play(Transform(R_mob, R_new), run_time=0.5)
                self.wait(0.15)
                self.play(FadeOut(cell_rect), run_time=0.25)

        self.play(FadeOut(r_group), FadeOut(expr), run_time=0.4)
        
        # B3: Đưa R về vị trí bên phải Q
        self.wait(0.5)
        self.play(R_group_mob.animate.to_edge(RIGHT, buff=1.25).shift(UP*0.25), run_time=0.8)

        self._clear_step()
        self._show_step("Bước 3: Kiểm tra lại")


        # Tạo mobject Q
        Q_mob2 = Matrix(np.round(Q_data, 2),left_bracket="(",right_bracket=")").scale(0.8).set_color(ORANGE)

        Q_label2 = MathTex("Q =", color=ORANGE).next_to(Q_mob2, LEFT, buff=0.3)
        Q_group2 = VGroup(Q_label2, Q_mob2)
        Q_group2.to_edge(LEFT, buff=1.25).shift(UP*0.3)

        self.play(Write(Q_label2), Create(Q_mob2), run_time=0.8)

        mul_sign = MathTex(r"\times", color=WHITE_C).scale(1.2)

        # Đưa dấu nhân vào giữa Q và R
        mul_sign.next_to(Q_group2, RIGHT, buff=0.8)
        self.play(FadeIn(mul_sign), run_time=0.5)

        # group Q và R 
        Q_R_Group = VGroup(Q_group2, mul_sign, R_group_mob)
        # Tính Q*R
        QR_data = Q_data @ R_data

        QR_mob = Matrix(np.round(QR_data, 2), left_bracket="(", right_bracket=")").scale(0.8).set_color(MINT)
        QR_label = MathTex("Q \\times R", color=MINT).next_to(QR_mob, DOWN, buff=0.25)
        QR_group_mob = VGroup(QR_label, QR_mob)

        # transform từ Q và R sang QR
        self.play(ReplacementTransform(Q_R_Group, QR_group_mob), run_time=1)

        # Đưa A vào phải QR
        A_mob2 = Matrix(A, left_bracket="(", right_bracket=")").scale(0.8).set_color(WHITE_C)
        A_label2 = Text("A", font=VFONT, font_size=24, color=WHITE_C, weight=BOLD).next_to(A_mob2, DOWN, buff=0.25)
        A_group2 = VGroup(A_label2, A_mob2)

        A_group2.next_to(QR_group_mob, RIGHT, buff=0.75).align_to(QR_group_mob, UP)
        self.play(Create(A_group2), run_time=0.8)

        # Đưa ký hiệu ≈ vào giữa QR và A
        approx2 = MathTex(r"\approx", color=WHITE_C).scale(1.2)
        approx2.next_to(A_group2, LEFT, buff=0.2)
        self.play(FadeIn(approx2), run_time=0.5)

        # Group lại QR và A để dễ dàng transform
        QR_A_Group = VGroup(QR_group_mob, approx2, A_group2)
        # Đưa QR_A_Group về giữa
        self.play(QR_A_Group.animate.center(), run_time=0.8)

        self.wait(1)

        concl = Text("Vậy: A ≈ Q × R", font=VFONT, font_size=26, color=GOLD, weight=BOLD)
        concl.to_edge(DOWN, buff=0.45)
        self.play(Write(concl), run_time=0.8)
        self.wait(2.0)

        self.play(
            FadeOut(QR_A_Group),
            FadeOut(concl),
            run_time=1
        )
        self._clear_step()


        # --- PHẦN 2: GIỚI THIỆU SVD ---
        title_svd = Text("Phân rã SVD: Singular Value Decomposition", font=VFONT, font_size=38, color=GOLD, weight=BOLD)
        sub_svd = Text("Mục tiêu: Phân rã ma trận A = U × Σ × Vᵀ", font=VFONT, font_size=26, color=CYAN).next_to(title_svd, DOWN)
        sub_svd2 = Text("U, Vᵀ: Phép xoay (Trực giao)  |  Σ: Phép co giãn (Đường chéo)", font=VFONT, font_size=22, color=MINT).next_to(sub_svd, DOWN, buff=0.5)
        
        self.play(Write(title_svd), run_time=1)
        self.play(Write(sub_svd), run_time=1)
        self.play(Write(sub_svd2), run_time=1)
        self.wait(1.5)
        self.play(FadeOut(title_svd), FadeOut(sub_svd), FadeOut(sub_svd2))

        A = np.array([[2.0, 1.0], 
                      [0.5, 2.0]])
        
        # Dùng numpy để tính SVD
        U, S, VT = np.linalg.svd(A)
        Sigma = np.diag(S)

        self._show_step("Ma trận A và các thành phần SVD")
        
        # Hiển thị A = U * Sigma * VT
        A_mob = Matrix(np.round(A, 2), left_bracket="(", right_bracket=")").set_color(WHITE_C)
        A_lbl = MathTex("A =", color=WHITE_C).next_to(A_mob, LEFT)
        A_group = VGroup(A_lbl, A_mob)

        eq = MathTex("=", color=WHITE_C)
        
        U_mob = Matrix(np.round(U, 2), left_bracket="(", right_bracket=")").set_color(CYAN)
        U_lbl = MathTex("U", color=CYAN).next_to(U_mob, DOWN)
        U_group = VGroup(U_mob, U_lbl)

        S_mob = Matrix(np.round(Sigma, 2), left_bracket="(", right_bracket=")").set_color(GOLD)
        S_lbl = MathTex(r"\Sigma", color=GOLD).next_to(S_mob, DOWN)
        S_group = VGroup(S_mob, S_lbl)

        VT_mob = Matrix(np.round(VT, 2), left_bracket="(", right_bracket=")").set_color(PINK)
        VT_lbl = MathTex("V^T", color=PINK).next_to(VT_mob, DOWN)
        VT_group = VGroup(VT_mob, VT_lbl)

        # Sắp xếp chúng thẳng hàng
        eq.next_to(A_group, RIGHT, buff=0.3)
        U_group.next_to(eq, RIGHT, buff=0.3)
        S_group.next_to(U_group, RIGHT, buff=0.2)
        VT_group.next_to(S_group, RIGHT, buff=0.2)
        
        full_eq_group = VGroup(A_group, eq, U_group, S_group, VT_group).center().scale(0.8)

        self.play(Create(A_group), run_time=1)
        self.play(FadeIn(eq), Create(U_group), Create(S_group), Create(VT_group), run_time=1)
        self.wait(1)
        
        self.play(full_eq_group.animate.scale(0.65).to_edge(UP, buff=0.8), run_time=1)
        self._clear_step()

        self._show_step("Trực quan hóa không gian: Vòng tròn đơn vị")
        
        plane = NumberPlane(
            x_range=[-4, 4], y_range=[-4, 4],
            background_line_style={"stroke_color": DIM, "stroke_opacity": 0.5}
        ).scale(0.6).to_edge(DOWN, buff=0.5)
        
        # Lấy tâm của mặt phẳng để làm mốc cho toàn bộ hệ thống
        center_pt = plane.get_center()
        
        # Vòng tròn đơn vị và vector dời về tâm mới
        circle = Circle(radius=0.6, color=WHITE_C).move_to(center_pt) 
        v_i = Arrow(center_pt, center_pt + np.array([0.6, 0, 0]), buff=0, color=CYAN, max_tip_length_to_length_ratio=0.15)
        v_j = Arrow(center_pt, center_pt + np.array([0, 0.6, 0]), buff=0, color=PINK, max_tip_length_to_length_ratio=0.15)
        
        lbl_i = MathTex(r"\hat{i}", color=CYAN)
        lbl_j = MathTex(r"\hat{j}", color=PINK)

        # Fix lại Updater để tính chuẩn vector hướng
        def update_label_i(m):
            vec = v_i.get_end() - center_pt # Vector hướng thực tế
            norm = np.linalg.norm(vec)
            if norm > 0: m.move_to(v_i.get_end() + (vec/norm) * 0.35)
                
        def update_label_j(m):
            vec = v_j.get_end() - center_pt # Vector hướng thực tế
            norm = np.linalg.norm(vec)
            if norm > 0: m.move_to(v_j.get_end() + (vec/norm) * 0.35)

        lbl_i.add_updater(update_label_i)
        lbl_j.add_updater(update_label_j)

        # Gom nhóm những thứ sẽ bị biến đổi
        moving_mobjects = VGroup(circle, v_i, v_j)

        self.play(Create(plane), run_time=1)
        self.play(Create(circle), Create(v_i), Create(v_j), FadeIn(lbl_i), FadeIn(lbl_j), run_time=1)
        self.wait(1)
        self._clear_step()

        # Xoay bằng V^T
        self._show_step("1. Nhân với Vᵀ: Xoay hệ trục (Rotation)")
        box_VT = SurroundingRectangle(VT_group, color=RED, buff=0.1)
        self.play(Create(box_VT), run_time=0.5)

        self.play(moving_mobjects.animate.apply_matrix(VT, about_point=center_pt), run_time=2)
        self.wait(1)
        self.play(FadeOut(box_VT))
        self._clear_step()

        # Co giãn bằng Sigma
        self._show_step("2. Nhân với Σ: Kéo giãn theo trục (Scaling)")
        box_S = SurroundingRectangle(S_group, color=RED, buff=0.1)
        self.play(Create(box_S), run_time=0.5)

        self.play(moving_mobjects.animate.apply_matrix(Sigma, about_point=center_pt), run_time=2)
        self.wait(1)
        self.play(FadeOut(box_S))
        self._clear_step()

        # Xoay bằng U
        self._show_step("3. Nhân với U: Xoay lần cuối (Rotation)")
        box_U = SurroundingRectangle(U_group, color=RED, buff=0.1)
        self.play(Create(box_U), run_time=0.5)

        self.play(moving_mobjects.animate.apply_matrix(U, about_point=center_pt), run_time=2)
        self.wait(1)
        self.play(FadeOut(box_U))
        self._clear_step()

        self._show_step("Toàn bộ quá trình tương đương với việc nhân ma trận A")
        box_A = SurroundingRectangle(A_group, color=RED, buff=0.1)
        self.play(Create(box_A), run_time=0.5)

        #Fix lại hình mờ
        orig_circle = Circle(radius=0.6, color=WHITE).set_opacity(0.2).move_to(center_pt)
        orig_vi = Arrow(center_pt, center_pt + np.array([0.6, 0, 0]), buff=0, color=WHITE).set_opacity(0.2)
        orig_vj = Arrow(center_pt, center_pt + np.array([0, 0.6, 0]), buff=0, color=WHITE).set_opacity(0.2)
        self.play(FadeIn(orig_circle), FadeIn(orig_vi), FadeIn(orig_vj), run_time=1)
        
        self.wait(2)

        lbl_i.clear_updaters()
        lbl_j.clear_updaters()
        self.play(*[FadeOut(m) for m in self.mobjects])

        self.wait(2)
    def _show_step(self, text):
        self._step_lbl = Text(text, font=VFONT, font_size=25, color=GOLD, weight=BOLD).to_edge(UP, buff=0.35)
        self._step_ul = Line(self._step_lbl.get_left(), self._step_lbl.get_right(), color=GOLD).next_to(self._step_lbl, DOWN, buff=0.1)
        self.play(Write(self._step_lbl), Create(self._step_ul), run_time=0.5)

    def _clear_step(self):
        self.play(FadeOut(self._step_lbl), FadeOut(self._step_ul), run_time=0.4)


# ─── BẢNG MÀU ──────────────────────────────────
BG       = "#0d1117"
WHITE_C  = "#e8eaf6"
BLUE     = "#4EA8DE"
CYAN     = "#56cfe1"
MINT     = "#72efdd"
GREEN    = "#57CC6E"
GOLD     = "#f0c060"
ORANGE   = "#F4A14B"
PINK     = "#ff6b9d"
RED      = "#FF6B6B"
LAVENDER = "#c77dff"
PURPLE   = "#B48FE0"
DIM      = "#8892a4"

VFONT    = "Arial"

class SVDCompressionScene(Scene):
    """
    Video 1: Ứng dụng SVD trong nén ảnh (Image Compression)
    """
    def construct(self):
        self.camera.background_color = BG

        # --- TIÊU ĐỀ ---
        title = Text("Ứng dụng SVD: Nén Ảnh", font=VFONT, font_size=42, color=GOLD, weight=BOLD)
        sub = Text("Ý tưởng: Cắt giữ lại k giá trị kỳ dị lớn nhất (Truncated SVD)", font=VFONT, font_size=24, color=CYAN).next_to(title, DOWN)
        self.play(Write(title), FadeIn(sub, shift=UP*0.2))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(sub))

        # --- BƯỚC 1: TRỰC QUAN HÓA MA TRẬN ẢNH VÀ SVD ĐẦY ĐỦ ---
        self._show_step("1. Phân rã ảnh (Ma trận A) thành U, Σ, Vᵀ")

        # Thay vì viết ma trận số, ta dùng các khối hình chữ nhật để biểu diễn kích thước
        # Ảnh gốc A (m x n), giả sử m=n=500
        A_rect = Rectangle(width=2.5, height=2.5, fill_color=BLUE, fill_opacity=0.3, stroke_color=BLUE)
        A_lbl = MathTex(r"A_{m \times n}").move_to(A_rect)
        A_group = VGroup(A_rect, A_lbl).to_edge(LEFT, buff=1.0)

        eq = MathTex("=").next_to(A_group, RIGHT, buff=0.3)

        # U (m x m)
        U_rect = Rectangle(width=2.5, height=2.5, fill_color=CYAN, fill_opacity=0.3, stroke_color=CYAN)
        U_lbl = MathTex(r"U_{m \times m}").move_to(U_rect)
        U_group = VGroup(U_rect, U_lbl).next_to(eq, RIGHT, buff=0.3)

        # Sigma (m x n) - Chỉ có đường chéo
        S_rect = Rectangle(width=2.5, height=2.5, fill_color=GOLD, fill_opacity=0.3, stroke_color=GOLD)
        S_diag = Line(S_rect.get_corner(UL), S_rect.get_corner(DR), color=GOLD, stroke_width=4)
        S_lbl = MathTex(r"\Sigma_{m \times n}").move_to(S_rect).shift(UR*0.5)
        S_group = VGroup(S_rect, S_diag, S_lbl).next_to(U_group, RIGHT, buff=0.2)

        # V^T (n x n)
        VT_rect = Rectangle(width=2.5, height=2.5, fill_color=PINK, fill_opacity=0.3, stroke_color=PINK)
        VT_lbl = MathTex(r"V^T_{n \times n}").move_to(VT_rect)
        VT_group = VGroup(VT_rect, VT_lbl).next_to(S_group, RIGHT, buff=0.2)

        full_svd = VGroup(A_group, eq, U_group, S_group, VT_group).center().shift(UP*0.5)

        self.play(FadeIn(A_group))
        self.play(Write(eq))
        self.play(FadeIn(U_group), FadeIn(S_group), FadeIn(VT_group), run_time=1.5)
        self.wait(1.5)

        memory_full = Text("Dung lượng gốc: m × n (VD: 500x500 = 250,000 pixels)", font=VFONT, font_size=20, color=WHITE_C).next_to(full_svd, DOWN, buff=1)
        self.play(Write(memory_full))
        self.wait(2)
        self._clear_step()

        # --- BƯỚC 2: TRUNCATED SVD ---
        self._show_step("2. Nén ảnh: Giữ lại k giá trị kỳ dị (k << m, n)")
        
        # Vẽ các nét đứt cắt ma trận
        k_width = 0.5 # Biểu diễn k

        # Cắt U (giữ lại k cột)
        U_cut = DashedLine(U_rect.get_corner(UL) + RIGHT*k_width, U_rect.get_corner(DL) + RIGHT*k_width, color=RED)
        U_keep = Rectangle(width=k_width, height=2.5, fill_color=CYAN, fill_opacity=0.7, stroke_color=RED).align_to(U_rect, UL)
        
        # Cắt Sigma (giữ lại k x k)
        S_cut_v = DashedLine(S_rect.get_corner(UL) + RIGHT*k_width, S_rect.get_corner(DL) + RIGHT*k_width, color=RED)
        S_cut_h = DashedLine(S_rect.get_corner(UL) + DOWN*k_width, S_rect.get_corner(UR) + DOWN*k_width, color=RED)
        S_keep = Rectangle(width=k_width, height=k_width, fill_color=GOLD, fill_opacity=0.7, stroke_color=RED).align_to(S_rect, UL)
        
        # Cắt VT (giữ lại k hàng)
        VT_cut = DashedLine(VT_rect.get_corner(UL) + DOWN*k_width, VT_rect.get_corner(UR) + DOWN*k_width, color=RED)
        VT_keep = Rectangle(width=2.5, height=k_width, fill_color=PINK, fill_opacity=0.7, stroke_color=RED).align_to(VT_rect, UL)

        self.play(Create(U_cut), Create(S_cut_v), Create(S_cut_h), Create(VT_cut), run_time=1)
        self.play(FadeIn(U_keep), FadeIn(S_keep), FadeIn(VT_keep))
        self.wait(1)

        # Làm mờ phần bị bỏ đi
        self.play(
            U_rect.animate.set_opacity(0.1), U_lbl.animate.set_opacity(0.1),
            S_rect.animate.set_opacity(0.1), S_diag.animate.set_opacity(0.1), S_lbl.animate.set_opacity(0.1),
            VT_rect.animate.set_opacity(0.1), VT_lbl.animate.set_opacity(0.1),
            run_time=1
        )

        # Di chuyển các khối k lại gần nhau
        U_k_lbl = MathTex("U_k", font_size=24).move_to(U_keep).rotate(PI/2)
        S_k_lbl = MathTex(r"\Sigma_k", font_size=24).move_to(S_keep)
        VT_k_lbl = MathTex("V_k^T", font_size=24).move_to(VT_keep)

        self.play(Write(U_k_lbl), Write(S_k_lbl), Write(VT_k_lbl))

        memory_k = Text("Dung lượng nén: k(m + n + 1). Rất nhỏ gọn!", font=VFONT, font_size=22, color=GREEN).next_to(memory_full, DOWN, buff=0.3)
        self.play(Write(memory_k))
        
        approx = MathTex(r"A \approx U_k \Sigma_k V_k^T", color=ORANGE, font_size=40).next_to(memory_k, DOWN, buff=0.5)
        self.play(Transform(eq, approx))
        
        self.wait(3)

    def _show_step(self, text):
        self._step_lbl = Text(text, font=VFONT, font_size=25, color=GOLD, weight=BOLD).to_edge(UP, buff=0.35)
        self._step_ul = Line(self._step_lbl.get_left(), self._step_lbl.get_right(), color=GOLD).next_to(self._step_lbl, DOWN, buff=0.1)
        self.play(Write(self._step_lbl), Create(self._step_ul), run_time=0.5)

    def _clear_step(self):
        self.play(FadeOut(self._step_lbl), FadeOut(self._step_ul), run_time=0.4)

class LUSolverScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Ứng dụng LU: Giải Hệ Phương Trình Tuyến Tính", font=VFONT, font_size=40, color=BLUE, weight=BOLD)
        sub = Text("Mục tiêu: Giải hệ Ax = b nhanh hơn nhiều lần khi b thay đổi liên tục", font=VFONT, font_size=22, color=CYAN).next_to(title, DOWN)
        self.play(Write(title), FadeIn(sub, shift=UP*0.2))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(sub))

        self._show_step("Ý tưởng thuật toán")

        eq1 = MathTex(r"A \mathbf{x} = \mathbf{b}", font_size=42, color=WHITE_C)
        
        eq2 = VGroup(
            MathTex(r"L U \mathbf{x} = \mathbf{b} \quad (", font_size=36, color=GOLD),
            Text("vì ", font=VFONT, font_size=24, color=GOLD),
            MathTex(r"A = LU)", font_size=36, color=GOLD)
        ).arrange(RIGHT, buff=0.1)
        
        step_y = VGroup(
            Text("1. Đặt ", font=VFONT, font_size=24, color=GREEN),
            MathTex(r"U\mathbf{x} = \mathbf{y} \Rightarrow ", color=GREEN, font_size=32),
            Text("Giải ", font=VFONT, font_size=24, color=GREEN),
            MathTex(r"L\mathbf{y} = \mathbf{b}", color=GREEN, font_size=32),
            Text(" (Thế tiến)", font=VFONT, font_size=24, color=GREEN)
        ).arrange(RIGHT, buff=0.1)

        step_x = VGroup(
            Text("2. Giải ", font=VFONT, font_size=24, color=CYAN),
            MathTex(r"U\mathbf{x} = \mathbf{y}", color=CYAN, font_size=32),
            Text(" (Thế lùi)", font=VFONT, font_size=24, color=CYAN)
        ).arrange(RIGHT, buff=0.1)

        theory_group = VGroup(eq1, eq2, step_y, step_x).arrange(DOWN, buff=0.4).shift(UP*1)
        
        self.play(Write(eq1))
        self.wait(0.5)
        self.play(FadeIn(eq2, shift=UP*0.2))
        self.wait(1)
        
        box1 = SurroundingRectangle(step_y, color=GREEN, buff=0.15)
        box2 = SurroundingRectangle(step_x, color=CYAN, buff=0.15)

        self.play(Write(step_y), Create(box1))
        self.wait(1)
        self.play(Write(step_x), Create(box2))
        self.wait(2)

        self.play(FadeOut(theory_group), FadeOut(box1), FadeOut(box2))
        self._clear_step()

        self._show_step("Ví dụ minh họa: Hệ 3x3")

        SC = 0.7
        L_val = [[1, 0, 0], [2, 1, 0], [-1, 0.5, 1]]
        U_val = [[3, -1, 2], [0, 4, 1], [0, 0, 2]]
        b_val = [[5], [14], [-2]]

        L_mat = Matrix(L_val).scale(SC).set_color(GREEN)
        U_mat = Matrix(U_val).scale(SC).set_color(CYAN)
        b_mat = Matrix(b_val).scale(SC).set_color(ORANGE)
        
        L_lbl = MathTex("L=").next_to(L_mat, LEFT)
        U_lbl = MathTex("U=").next_to(U_mat, LEFT)
        b_lbl = MathTex(r"\mathbf{b}=").next_to(b_mat, LEFT)

        mat_group = VGroup(
            VGroup(L_lbl, L_mat),
            VGroup(U_lbl, U_mat),
            VGroup(b_lbl, b_mat)
        ).arrange(RIGHT, buff=1.0).to_edge(UP, buff=1.5)

        self.play(FadeIn(mat_group, shift=UP*0.3))
        self.wait(1)

        lbl_step1 = VGroup(
            Text("1) Giải ", font=VFONT, font_size=32, color=GREEN),
            MathTex(r"L\mathbf{y} = \mathbf{b}", color=GREEN)
        ).arrange(RIGHT, buff=0.15).move_to(LEFT*3.5 + DOWN*0.5)
        self.play(Write(lbl_step1))

        L_calc = Matrix(L_val).scale(SC).set_color(GREEN)
        y_mat = Matrix([["y_1"], ["y_2"], ["y_3"]]).scale(SC).set_color(WHITE_C)
        eq_calc1 = MathTex("=")
        b_calc = Matrix(b_val).scale(SC).set_color(ORANGE)

        calc_g1 = VGroup(L_calc, y_mat, eq_calc1, b_calc).arrange(RIGHT, buff=0.2).next_to(lbl_step1, DOWN, buff=0.3).align_to(lbl_step1, LEFT)
        self.play(FadeIn(calc_g1))
        
        y_res_val = [[5], [4], [-1]] 
        y_res_mat = Matrix(y_res_val).scale(SC).set_color(WHITE_C)
        y_res_lbl = MathTex(r"\Rightarrow \mathbf{y} =").set_color(GREEN)
        res_g1 = VGroup(y_res_lbl, y_res_mat).arrange(RIGHT, buff=0.2).next_to(calc_g1, RIGHT, buff=0.8)
        
        self.play(FadeIn(res_g1, shift=LEFT*0.3))
        box_y = SurroundingRectangle(res_g1, color=GREEN)
        self.play(Create(box_y))
        self.wait(1.5)

        self.play(
            FadeOut(lbl_step1, shift=UP*0.5),
            FadeOut(calc_g1, shift=UP*0.5),
            FadeOut(res_g1, shift=UP*0.5),
            FadeOut(box_y, shift=UP*0.5),
            run_time=1
        )

        lbl_step2 = VGroup(
            Text("2) Giải ", font=VFONT, font_size=32, color=CYAN),
            MathTex(r"U\mathbf{x} = \mathbf{y}", color=CYAN)
        ).arrange(RIGHT, buff=0.15).move_to(LEFT*3.5 + DOWN*0.5) # Vị trí y hệt Bước 1
        self.play(Write(lbl_step2))

        U_calc = Matrix(U_val).scale(SC).set_color(CYAN)
        x_mat = Matrix([["x_1"], ["x_2"], ["x_3"]]).scale(SC).set_color(PINK)
        eq_calc2 = MathTex("=")
        y_calc = Matrix(y_res_val).scale(SC).set_color(WHITE_C)

        calc_g2 = VGroup(U_calc, x_mat, eq_calc2, y_calc).arrange(RIGHT, buff=0.2).next_to(lbl_step2, DOWN, buff=0.3).align_to(lbl_step2, LEFT)
        self.play(FadeIn(calc_g2))

        x_res_val = [["2.375"], ["1.125"], ["-0.5"]]
        x_res_mat = Matrix(x_res_val).scale(SC).set_color(PINK)
        x_res_lbl = MathTex(r"\Rightarrow \mathbf{x} =").set_color(CYAN)
        res_g2 = VGroup(x_res_lbl, x_res_mat).arrange(RIGHT, buff=0.2).next_to(calc_g2, RIGHT, buff=0.8)

        self.play(FadeIn(res_g2, shift=LEFT*0.3))
        box_x = SurroundingRectangle(res_g2, color=CYAN)
        self.play(Create(box_x))
        self.wait(2)

        self._clear_step()
        self.play(FadeOut(mat_group, shift=UP*0.5)) 

        step2_group = VGroup(lbl_step2, calc_g2, res_g2, box_x)
        self.play(step2_group.animate.center().shift(UP*0.5))
        
        concl = Text("Độ phức tạp giảm từ O(n³) xuống O(n²)", font=VFONT, font_size=30, color=GOLD)
        concl.next_to(step2_group, DOWN, buff=1.0)
        self.play(Write(concl))
        
        self.wait(3)

    def _show_step(self, text):
        self._step_lbl = Text(text, font=VFONT, font_size=25, color=GOLD, weight=BOLD).to_edge(UP, buff=0.35)
        self._step_ul = Line(self._step_lbl.get_left(), self._step_lbl.get_right(), color=GOLD).next_to(self._step_lbl, DOWN, buff=0.1)
        self.play(Write(self._step_lbl), Create(self._step_ul), run_time=0.5)

    def _clear_step(self):
        self.play(FadeOut(self._step_lbl), FadeOut(self._step_ul), run_time=0.4)


class FullPresentation(LUDecomposition, SVD, DiagonalizationScene, SVDCompressionScene, LUSolverScene):
    def construct(self):
        # 1. LU
        LUDecomposition.construct(self)
        self.clear()
        
        # 2. QR and SVD
        SVD.construct(self)
        self.clear()
        
        # 3. Diagonalization
        DiagonalizationScene.construct(self)
        self.clear()
        
        # 4. SVD Compression
        SVDCompressionScene.construct(self)
        self.clear()
        
        # 5. LU Solver
        LUSolverScene.construct(self)