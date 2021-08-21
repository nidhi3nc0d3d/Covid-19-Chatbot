from PIL import ImageGrab


def press_ss_for_screenshot(count):
    while True:
        try:
            image = ImageGrab.grab(bbox=(100, 200, 1200, 950))
            image.save(f"screenshot{count}.png")
            print(f"Screenshot name = screenshot{count}.")
            break
        except:
            break
