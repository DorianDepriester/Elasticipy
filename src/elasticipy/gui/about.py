from qtpy.QtWidgets import QVBoxLayout, QLabel, QPushButton
from qtpy.QtGui import QPixmap
from qtpy.QtCore import Qt
from importlib import resources

def about(dialog):
    dialog.setWindowTitle("About Elasticipy")
    dialog.setFixedWidth(400)

    layout = QVBoxLayout(dialog)

    logo = QLabel()
    pixmap = QPixmap(
        str(resources.files("elasticipy.resources") / "logo_text.png")
    )
    pixmap = pixmap.scaledToWidth(250, Qt.SmoothTransformation)
    logo.setPixmap(pixmap)
    logo.setAlignment(Qt.AlignCenter)
    layout.addWidget(logo)

    # --- Text ---
    text = QLabel(
        "A Python library for elasticity tensors computations<br><br>"
        "© 2025–2026 Dorian Depriester, MIT Licence"
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