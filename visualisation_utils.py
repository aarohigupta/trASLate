import matplotlib.pyplot as plt
import numpy as np

def get_image(letter: str, data):
    # map the letter to the index in the dataframe
    letter_index = ord(letter.lower()) - 97
    # get the image data
    image = data[data['label'] == letter_index].iloc[0, 1:].values
    # reshape the image data into a 28x28 image
    image = image.reshape(28, 28)
    return image

def plot_image(letter: str, data):
    if letter == 'j' or letter == 'z':
        print(f'letter {letter} not found')
        return
    image = get_image(letter, data)
    if image.all() == None:
        print(f'letter {letter} not found')
        return
    # plot the image
    plt.imshow(image, cmap='gray')
    plt.title(f'Letter: {letter.upper()}')
    plt.show()

def plot_images(data):
    letters = 'abcdefghiklmnopqrstuvwxy'
    # plot the images in a grid
    fig, axes = plt.subplots(4, 6, figsize=(16, 16))
    for i, letter in enumerate(letters):
        image = get_image(letter, data)
        # plot the image
        axes[i // 6, i % 6].imshow(image, cmap='gray')
        axes[i // 6, i % 6].set_title(f'Letter: {letter.upper()}')