import matplotlib.pyplot as plt
from PIL import Image

def create_dashboard():

    original = Image.open("images/test.jpg")
    lime = Image.open("outputs/lime_result.png")

    fig, ax = plt.subplots(1,2, figsize=(10,5))

    ax[0].imshow(original)
    ax[0].set_title("Original Image")

    ax[1].imshow(lime)
    ax[1].set_title("LIME Explanation")

    for a in ax:
        a.axis("off")

    plt.savefig("outputs/comparison.png")
    print("Comparison dashboard created.")