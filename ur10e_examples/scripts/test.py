import cv2

def show_channel(input_image_path, channel='blue'):
    # Read the image
    image = cv2.imread(input_image_path)

    # Split the image into channels
    b, g, r = cv2.split(image)

    # Select the desired channel
    if channel.lower() == 'red':
        channel_img = r
    elif channel.lower() == 'green':
        channel_img = g
    else:  # Default to blue channel
        channel_img = b

    # Display the selected channel
    cv2.imshow(f'{channel.capitalize()} Channel', channel_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
image_path = '/dev_ws/src/ur10e_examples/scripts/label_02.jpeg' 
show_channel(image_path, 'blue')
