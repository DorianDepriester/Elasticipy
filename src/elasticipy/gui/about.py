from qtpy.QtWidgets import QVBoxLayout, QLabel, QPushButton
from qtpy.QtGui import QPixmap
from qtpy.QtCore import Qt

def about(dialog, logo_path):
    dialog.setWindowTitle("About elasticipy")
    dialog.setFixedWidth(400)

    layout = QVBoxLayout(dialog)

    if logo_path.exists():
        logo = QLabel()
        pixmap = QPixmap(str(logo_path))
        pixmap = pixmap.scaledToWidth(250, Qt.SmoothTransformation)
        logo.setPixmap(pixmap)
        logo.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo)

    # --- Text ---
    text = QLabel(
        "A Python library for elasticity tensors computations<br><br>"
        "© 2024–2025 Dorian Depriester, MIT Licence"
    )
    text.setAlignment(Qt.AlignCenter)
    layout.addWidget(text)

    # --- Link ---
    link = QLabel(
        '<a href="https://elasticipy.readthedocs.io/">'
        'https://elasticipy.readthedocs.io/</a>'
    )
    link.setAlignment(Qt.AlignCenter)
    link.setOpenExternalLinks(True)
    layout.addWidget(link)

    # --- Bug report ---
    link = QLabel(
        '<a href="https://github.com/DorianDepriester/Elasticipy/issues">'
        'Report a bug</a>'
    )
    link.setAlignment(Qt.AlignCenter)
    link.setOpenExternalLinks(True)
    layout.addWidget(link)

    # --- Close button ---
    close_btn = QPushButton("Close")
    close_btn.clicked.connect(dialog.close)
    layout.addWidget(close_btn)
    return dialog