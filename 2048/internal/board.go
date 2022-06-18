package internal

import (
	"fmt"
	"github.com/fatih/color"
	"math/rand"
	"reflect"
	"time"
)

const (
	_rows = 4
	_cols = 4

	probabilitySpace = 100
	probabilityOfTwo = 80 // probabilityOfTwo times 2 will come as new element out of  probabilitySpace1
)

type IBoard interface {
	Display() string
	AddElement()
	IsOver() bool
	CountScore() (int32, int32)
	Move(dir Dir)
}

type Board struct {
	Matrix [][]int
	over   bool
	newRow int
	newCol int
	IBoard
}

func (b *Board) CountScore() (int32, int32) {
	total := int32(0)
	maximum := int32(0)
	matrix := b.Matrix
	for i := 0; i < _rows; i++ {
		for j := 0; j < _cols; j++ {
			total += int32(matrix[i][j])
			maximum = int32(max(int(maximum), matrix[i][j]))
		}
	}
	return maximum, total
}

func max(one int, two int) int {
	if one > two {
		return one
	}
	return two
}

func (b *Board) IsOver() bool {
	empty := 0
	for i := 0; i < _rows; i++ {
		for j := 0; j < _cols; j++ {
			if b.Matrix[i][j] == 0 {
				empty++
			}
		}
	}
	return empty == 0 || b.over
}

// AddElement : it first finds the empty slots in the Board. They are the one with 0 value
// The it places a new cell randomly in one of those empty places
// The new value to put is also calculated randomly
func (b *Board) AddElement() {
	s1 := rand.NewSource(time.Now().UnixNano())
	r1 := rand.New(s1)
	val := r1.Int() % probabilitySpace
	if val <= probabilityOfTwo {
		val = 2
	} else {
		val = 4
	}

	empty := 0
	for i := 0; i < _rows; i++ {
		for j := 0; j < _cols; j++ {
			if b.Matrix[i][j] == 0 {
				empty++
			}
		}
	}
	elementCount := r1.Int()%empty + 1
	index := 0

	for i := 0; i < _rows; i++ {
		for j := 0; j < _cols; j++ {
			if b.Matrix[i][j] == 0 {
				index++
				if index == elementCount {
					b.newRow = i
					b.newCol = j
					b.Matrix[i][j] = val
					return
				}
			}
		}
	}
	return
}

func (b *Board) Display() string {
	d := color.New(color.FgBlue, color.Bold)

	var res = ""

	//b.Matrix = getRandom()
	for i := 0; i < len(b.Matrix); i++ {
		res += printHorizontal()
		res += "|"

		for j := 0; j < len(b.Matrix[0]); j++ {

			res += fmt.Sprintf("%3s", "")

			if b.Matrix[i][j] == 0 {
				res += fmt.Sprintf("%-6s|", "")
			} else {
				if i == b.newRow && j == b.newCol {
					res += d.Sprintf("%-6d|", b.Matrix[i][j])
				} else {
					res += fmt.Sprintf("%-6d|", b.Matrix[i][j])
				}
			}
		}
		res += fmt.Sprintf("%4s", "")
		res += "\n"
	}
	res += printHorizontal()
	return res
}

// printHorizontal prints a grid row
func printHorizontal() string {
	var res = ""
	for i := 0; i < 40; i++ {
		res += "-"
	}
	return res + "\n"
}

func New() Board {
	matrix := make([][]int, 0)
	for i := 0; i < _rows; i++ {
		matrix = append(matrix, make([]int, _cols))
	}
	return Board{
		Matrix: matrix,
	}
}

func (b *Board) AllMoves() []Dir {
	var res []Dir
	if b.isMoveAvailable(UP) {
		res = append(res, UP)
	}
	if b.isMoveAvailable(DOWN) {
		res = append(res, DOWN)
	}
	if b.isMoveAvailable(LEFT) {
		res = append(res, LEFT)
	}
	if b.isMoveAvailable(RIGHT) {
		res = append(res, RIGHT)
	}
	return res
}

func (b *Board) isMoveAvailable(dir Dir) bool {
	clone := b.clone()
	clone.Move(dir)
	return !reflect.DeepEqual(flattenMatrix(b.Matrix), flattenMatrix(clone.Matrix))
}

func (b *Board) clone() Board {
	return Board{Matrix: b.Matrix}
}
