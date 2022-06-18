package internal

type Dir int

const (
	UP Dir = iota
	DOWN
	LEFT
	RIGHT
	NO_DIR
)

// Move : this is the function which takes care of the movement of the SBoard when a key is pressed
// Move left is the lowest level function which is implemented
// other directions use Move left for their implementations
func (b *SBoard) Move(dir Dir) {
	switch dir {
	case LEFT:
		b.moveLeft()
	case RIGHT:
		b.moveRight()
	case DOWN:
		b.moveDown()
	case UP:
		b.moveUp()
	}
}

func (b *SBoard) moveLeft() {
	for i := 0; i < _rows; i++ {
		old := b.Matrix[i]
		b.Matrix[i] = movedRow(old)
	}
}

func (b *SBoard) moveUp() {
	b.reverseRows()
	b.moveDown()
	b.reverseRows()
}

func (b *SBoard) moveDown() {
	b.transpose()
	b.moveLeft()
	b.transpose()
	b.transpose()
	b.transpose()
}

func (b *SBoard) moveRight() {
	b.reverse()
	b.moveLeft()
	b.reverse()
}

// movedRow simply finds empty elements and filled elements
// it places the filled element in the beginning of the row
// [2 0 3 0] will become [2 3 0 0]
// an empty cell is displayed with 0 value
func movedRow(elems []int) []int {
	nonEmpty := make([]int, 0)
	for i := 0; i < _cols; i++ {
		if elems[i] != 0 {
			nonEmpty = append(nonEmpty, elems[i])
		}
	}
	remaining := _cols - len(nonEmpty)
	for i := 0; i < remaining; i++ {
		nonEmpty = append(nonEmpty, 0)
	}
	return mergeElements(nonEmpty)
}

// reverse simply reverses each row of the SBoard
func (b *SBoard) reverse() {
	for i := 0; i < _rows; i++ {
		b.Matrix[i] = reverseRow(b.Matrix[i])
	}
}

// transpose rotates a list
// row becomes _cols
// [ 1 2 ]
// [ 3 4 ] becomes
//
// [ 3 1 ]
// [ 4 2 ]
// see test for more clarity
func (b *SBoard) transpose() {
	ans := make([][]int, 0)
	for i := 0; i < _rows; i++ {
		ans = append(ans, make([]int, _cols))
	}
	for i := 0; i < _rows; i++ {
		for j := 0; j < _cols; j++ {
			ans[i][j] = b.Matrix[_cols-j-1][i]
		}
	}
	b.Matrix = ans
}

// reverseRows reverses the order of lists
// [1 2]
// [3 4] becomes
//
// [3 4]
// [1 2]
func (b *SBoard) reverseRows() {
	ans := make([][]int, 0)
	for i := 0; i < _rows; i++ {
		ans = append(ans, make([]int, _cols))
	}
	for i := 0; i < _rows; i++ {
		for j := 0; j < _cols; j++ {
			ans[_rows-i-1][j] = b.Matrix[i][j]
		}
	}
	b.Matrix = ans
}

// reverseRow reverses a row
func reverseRow(arr []int) []int {
	ans := make([]int, 0)
	for i := len(arr) - 1; i >= 0; i-- {
		ans = append(ans, arr[i])
	}
	return ans
}

// mergeElements when a row is moved to left, it merges the element which can
// see tests for more clarity
func mergeElements(arr []int) []int {
	newArr := make([]int, len(arr))
	newArr[0] = arr[0]
	index := 0
	for i := 1; i < len(arr); i++ {
		if arr[i] == newArr[index] {
			newArr[index] += arr[i]
		} else {
			index++
			newArr[index] = arr[i]
		}
	}
	return newArr
}
