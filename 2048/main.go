package main

import (
	"fmt"
	"github.com/aaarrti/RL-2048/2048/internal"
	pb "github.com/aaarrti/RL-2048/proto/go"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
	"log"
	"net"
)

func main() {

	addr := fmt.Sprintf("0.0.0.0:%v", internal.MainPort)
	listener, err := net.Listen("tcp", addr)
	internal.Must(err)

	//logger := util2.CreateLogger()

	log.Printf("----> GRPC listeninng on %v\n\n", addr)

	_server := grpc.NewServer(
	//grpc.UnaryInterceptor(grpczap.UnaryServerInterceptor(logger)),
	)
	pb.RegisterEnvServiceServer(_server, &internal.GameServer{})
	reflection.Register(_server)
	err = _server.Serve(listener)
	internal.Must(err)
}
