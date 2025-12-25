from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

output_path = "reports/HMM_Regime_Strategy_Research_Note.pdf"

doc = SimpleDocTemplate(
    output_path,
    pagesize=A4,
    rightMargin=2*cm,
    leftMargin=2*cm,
    topMargin=2*cm,
    bottomMargin=2*cm
)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="MyTitle",
    fontSize=18,
    leading=22,
    spaceAfter=20,
    fontName="Helvetica-Bold"
))
styles.add(ParagraphStyle(
    name="Section",
    fontSize=14,
    leading=18,
    spaceBefore=16,
    spaceAfter=10,
    fontName="Helvetica-Bold"
))
styles.add(ParagraphStyle(
    name="Body",
    fontSize=11,
    leading=14,
    spaceAfter=8
))

story = []

story.append(Paragraph("Hidden Markov Regime Strategy", styles["MyTitle"]))


story.append(Paragraph(
    "Abstract<br/>"
    "This research presents a regime-aware quantitative trading strategy "
    "based on Hidden Markov Models (HMM). The focus is on statistical "
    "robustness, out-of-sample validation, and downside risk control rather "
    "than raw return maximization.",
    styles["Body"]
))

story.append(Spacer(1, 12))

sections = {
    "Methodology": (
        "A multivariate Gaussian HMM is trained using an expanding walk-forward "
        "framework to avoid look-ahead bias. Market regimes are inferred daily "
        "and mapped to expected Sharpe ratios, which are converted into "
        "continuous portfolio exposure via a sigmoid function."
    ),
    "Risk Management": (
        "The strategy integrates a trend filter based on exponential moving "
        "averages and volatility targeting to normalize risk across time."
    ),
    "Validation": (
        "Robustness is assessed using block bootstrap Monte Carlo simulations "
        "and the Deflated Sharpe Ratio, accounting for non-normal returns and "
        "data-snooping effects."
    ),
    "Results": (
        "While raw returns underperform Buy & Hold, volatility-matched "
        "performance is comparable, with significantly reduced drawdowns "
        "during crisis periods."
    ),
    "Conclusion": (
        "The strategy demonstrates that regime-aware exposure control can "
        "deliver statistically defensible performance with superior risk "
        "characteristics."
    )
}

for title, text in sections.items():
    story.append(Paragraph(title, styles["Section"]))
    story.append(Paragraph(text, styles["Body"]))

doc.build(story)

print(f"PDF generado en: {output_path}")
