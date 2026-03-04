import qrcode

# 1. Define la URL de tu vídeo
video_url = "https://youtu.be/G2IWO-ttQWU?si=cvCTkC6ogSM3noOj"

# 2. Genera el QR
qr = qrcode.QRCode(
    version=2,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)
qr.add_data(video_url)
qr.make(fit=True)

# 3. Crea la imagen
img = qr.make_image(fill_color="black", back_color="white")
img.save("qr_mario_video.png")
