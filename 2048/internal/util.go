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

func mapMove(enum pb.MoveEnum) Dir {
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
		return NO_DIR
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
