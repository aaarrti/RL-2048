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
	game     Board
	maxScore int
	ogMatrix [][]int
}

func NewServerGame() ServerGameType {
	b := New()
	b.AddElement()
	b.AddElement()
	return ServerGameType{game: b, maxScore: 0,
		ogMatrix: b.Matrix,
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
	move := enumToDir(in.Value)
	s.game.Move(move)

	scoreMax, ScoreTotal := s.game.CountScore()

	res := pb.GameState{
		Value:          flattenMatrix(s.game.Matrix),
		MovesAvailable: !s.game.IsOver(),
		Score:          &pb.ScoreTuple{Max: scoreMax, Total: ScoreTotal},
	}
	return &res, nil
}

func (s *ServerGameType) Reset(context.Context, *emptypb.Empty) (*pb.GameState, error) {
	s.game.Matrix = s.ogMatrix
	return &pb.GameState{Value: flattenMatrix(s.ogMatrix)}, nil
}

func (s *ServerGameType) AvailableMoves(context.Context, *emptypb.Empty) (*pb.AllMoves, error) {

	res := s.game.AllMoves()
	out := pb.AllMoves{}

	for _, i := range res {
		out.Moves = append(out.Moves, dirToEnum(i))
	}
	return &out, nil
}

func (s *ServerGameType) Render(context.Context, *emptypb.Empty) (*pb.StringMessage, error) {
	view := s.game.Display()
	fmt.Println(view)
	return &pb.StringMessage{Value: view}, nil
}
