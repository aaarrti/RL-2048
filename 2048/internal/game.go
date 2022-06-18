package internal

import (
	"context"
	"fmt"
	pb "github.com/aaarrti/RL-2048/proto/go"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
	"google.golang.org/protobuf/types/known/emptypb"
	"log"
	"net"
)

type ServerGameType struct {
	pb.GameServiceServer
	game     IBoard
	maxScore int
	ogMatrix [][]int
}

func NewServerGame() ServerGameType {
	b := New()
	b.AddElement()
	b.AddElement()
	return ServerGameType{game: b, maxScore: 0,
		ogMatrix: b.(*SBoard).Matrix,
	}
}

func (s *ServerGameType) ServerGame(port int) {
	addr := fmt.Sprintf("0.0.0.0:%v", port)
	listener, err := net.Listen("tcp", addr)
	Must(err)

	//logger := util.CreateLogger()

	log.Printf("----> GRPC serving Game on %v\n\n", addr)

	_server := grpc.NewServer(
	//grpc.UnaryInterceptor(grpczap.UnaryServerInterceptor(logger)),
	)
	pb.RegisterGameServiceServer(_server, s)
	reflection.Register(_server)
	err = _server.Serve(listener)
	Must(err)
}

func (s *ServerGameType) DoMove(ctx context.Context, in *pb.MoveMessage) (*pb.GameState, error) {
	//fmt.Printf("Received Move: %v\n", in.Value)
	move := mapMove(in.Value)
	s.game.Move(move)

	res := pb.GameState{
		Value:          flattenMatrix(s.game.(*SBoard).Matrix),
		MovesAvailable: !s.game.IsOver(),
	}
	return &res, nil
}

func (s *ServerGameType) Reset(context.Context, *emptypb.Empty) (*pb.GameState, error) {
	s.game.(*SBoard).Matrix = s.ogMatrix
	return &pb.GameState{Value: flattenMatrix(s.ogMatrix)}, nil
}
