def print_banner(text: str, border_char: str = "*", width: int = 60):
    """
    Print a centered banner with a border.

    :param text: The text to display in the banner
    :param border_char: Character used for the border
    :param width: Total width of the banner
    """
    if len(text) + 4 > width:
        width = len(text) + 4  # ensure text fits

    print(border_char * width)
    print(border_char + " " * ((width - 2 - len(text)) // 2) + text + " " * ((width - 2 - len(text) + 1) // 2) + border_char)
    print(border_char * width)