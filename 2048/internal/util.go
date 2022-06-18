package internal

import (
	pb "github.com/aaarrti/RL-2048/proto/go"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"log"
	"time"
)

func Must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func CreateLogger() *zap.Logger {
	loggerConfig := zap.NewDevelopmentConfig()
	loggerConfig.DisableStacktrace = true
	loggerConfig.EncoderConfig.TimeKey = "timestamp"
	loggerConfig.EncoderConfig.EncodeTime = zapcore.TimeEncoderOfLayout(time.RFC3339)

	_logger, err := loggerConfig.Build()
	Must(err)
	return _logger
}

func enumToDir(enum pb.MoveEnum) Dir {
	switch enum {
	case pb.MoveEnum_UP:
		return UP
	case pb.MoveEnum_DOWN:
		return DOWN
	case pb.MoveEnum_LEFT:
		return LEFT
	case pb.MoveEnum_RIGHT:
		return RIGHT
	default:
		log.Fatalf("Unknown move %v\n", enum)
		return -1
	}
}

func dirToEnum(dir Dir) pb.MoveEnum {
	switch dir {
	case UP:
		return pb.MoveEnum_UP
	case DOWN:
		return pb.MoveEnum_DOWN
	case LEFT:
		return pb.MoveEnum_LEFT
	case RIGHT:
		return pb.MoveEnum_RIGHT
	default:
		log.Fatalf("Unknown move %v\n", dir)
		return -1
	}
}

func flattenMatrix(matrix [][]int) []int32 {
	var flatArr []int32
	for _, row := range matrix {
		for _, i := range row {
			flatArr = append(flatArr, int32(i))
		}
	}
	return flatArr
}
