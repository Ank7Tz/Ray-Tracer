package main

import (
	"context"
	"flag"
	"log"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/Ank7Tz/Ray-Tracer/proto"
)

var (
	addr = flag.String("addr", "localhost:50051", "Server Ip address and port")
)

func main() {
	conn, err := grpc.NewClient(*addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()

	c := pb.NewWorldDataServiceClient(conn)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	r, err := c.RegisterClient(ctx, &pb.ClientInfo{
		ClientId: "1",
	})
	if err != nil {
		log.Fatalf("could not register with server: %v", err)
	}

	log.Printf("response: %v", r)

}
