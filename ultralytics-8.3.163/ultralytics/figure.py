import matplotlib.pyplot as plt

# ====== 数据 ======
labels = ["Full", "w/o ReID", "w/o relink", "w/o OBB"]
idf1   = [0.6762, 0.5907, 0.3448, 0.5093]

# ====== 画图 ======
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)

bars = ax.bar(labels, idf1, width=0.55, color="#8FA8FF", alpha=0.85, edgecolor="black", linewidth=0.8)

# 坐标轴与网格
ax.set_ylabel("IDF1")
ax.set_ylim(0, 0.75)
ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
ax.set_axisbelow(True)

# 每个柱子顶部写数值
for b, v in zip(bars, idf1):
    ax.text(
        b.get_x() + b.get_width() / 2,
        v + 0.015,              # 往上抬一点
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

# 让排版更紧凑、避免遮挡
plt.xticks(rotation=0)
plt.tight_layout()

# ====== 导出 ======
plt.savefig("ablation_idf1.png", bbox_inches="tight")
plt.savefig("ablation_idf1.pdf", bbox_inches="tight")  # 论文更推荐 PDF 矢量图
plt.show()
