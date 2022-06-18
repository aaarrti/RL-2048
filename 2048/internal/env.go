package internal

import (
	"context"
	pb "github.com/aaarrti/RL-2048/proto/go"
	"google.golang.org/protobuf/types/known/emptypb"
)

type GameServer struct {
	pb.EnvServiceServer
}

var portGenerator = MainPort

func (c *GameServer) ProvisionEnvironment(context.Context, *emptypb.Empty) (*pb.IntMessage, error) {
	portGenerator++
	go func() {
		s := NewServerGame()
		s.ServerGame(portGenerator)
	}()
	return &pb.IntMessage{Value: int32(portGenerator)}, nil
}
