from typing import List


def save_text_array(array: List[str], path: str) -> None:
	with open(path, 'w', encoding = 'utf-8') as f:
		f.truncate(0)

		for i in range(len(array)):
			f.write(array[i])
			if i != len(array) - 1:
				f.write('\n')


def load_text_array(path: str) -> List[str]:
	with open(path, 'r', encoding = 'utf-8') as f:

		return f.read().split('\n')
