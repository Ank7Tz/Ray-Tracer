package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net"
	"os"
	"strconv"
	"strings"

	pb "github.com/Ank7Tz/Ray-Tracer/proto"

	"google.golang.org/grpc"
)

var (
	port = flag.Int("port", 50051, "The server port")
)

type server struct {
	pb.UnimplementedWorldDataServiceServer
}

func cmdListener() {
	var cmd string
	var start_range int
	var end_range int
	// var hittable_list *pb.WorldData
	var world []Sphere
	scanner := bufio.NewScanner(os.Stdin)

	// default values
	start_range = -2
	end_range = 2
	for {
		fmt.Print(">> ")
		scanner.Scan()

		cmd = scanner.Text()

		switch {
		case strings.HasPrefix(cmd, "set"):
			cmd = strings.TrimSpace(cmd)
			token := strings.Split(cmd, " ")
			if len(token) == 1 || len(token) != 3 {
				fmt.Println("usage - set start_range=10 end_range=20")
				continue
			}
			start := strings.Split(token[1], "=")
			end := strings.Split(token[2], "=")
			if len(start) != 2 || len(end) != 2 {
				fmt.Println("usage - set start_range=10 end_range=20")
				continue
			}
			var err error
			if start_range, err = strconv.Atoi(start[1]); err != nil {
				fmt.Println("usage - set start_range=10 end_range=20")
				continue
			}

			if end_range, err = strconv.Atoi(end[1]); err != nil {
				fmt.Println("usage - set start_range=10 end_range=20")
				continue
			}

			fmt.Printf("start: %d \t end: %d\n", start_range, end_range)

		case strings.EqualFold(cmd, "generate"):
			world = generateWorld(start_range, end_range)
			fmt.Printf("world: %v\n", world)
		case strings.EqualFold(cmd, "exit"):
			fmt.Println("closing...")
			return
		default:
			fmt.Println("What is this???")
		}
	}
}

type Color [3]float32
type Vec3 [3]float32
type World []Sphere

func (w World) String() string {
	var result strings.Builder

	result.WriteString("world: [")

	for i := 0; i < len(w); i++ {
		result.WriteString(fmt.Sprintf("%v, ", w[0]))
	}

	result.WriteString("]")

	return result.String()
}

func (c Color) String() string {
	return fmt.Sprintf("color: [%f, %f, %f]", c[0], c[1], c[2])
}

func (v Vec3) String() string {
	return fmt.Sprintf("vec3: [%f, %f, %f]", v[0], v[1], v[2])
}

func randomColor() Color {
	return Color{rand.Float32(), rand.Float32(), rand.Float32()}
}

type Material interface {
	String() string
}

type MaterialBase struct{}

type Metal struct {
	MaterialBase
	Albedo Color
	Fuzz   float32
}

func (m Metal) String() string {
	return fmt.Sprintf("[Type: Metal, Albedo: %v, Fuzz: %f]", m.Albedo, m.Fuzz)
}

type Dielectric struct {
	MaterialBase
	Refraction_index float32
}

func (d Dielectric) String() string {
	return fmt.Sprintf("[Type: Dielectric, Refraction_index: %f]", d.Refraction_index)
}

type Lambertian struct {
	MaterialBase
	Albedo Color
}

func (l Lambertian) String() string {
	return fmt.Sprintf("[Type: Lambertian, Albedo: %v]", l.Albedo)
}

type Sphere struct {
	Center Vec3
	Radius float32
	Mat    Material
}

func (s Sphere) String() string {
	return fmt.Sprintf("Sphere: [Center: %v, Radius: %f, Material: %v]", s.Center, s.Radius, s.Mat)
}

func generateSphere(a float32, b float32, height float32) *Sphere {
	id := rand.Intn(3)
	var material Material
	sphere := new(Sphere)

	switch id {
	case 0:
		// lambertian
		c := randomColor()
		mat := new(Lambertian)
		mat.Albedo = c
		material = mat
	case 1:
		// metal
		c := randomColor()
		fuzz := rand.Float32()
		mat := new(Metal)
		mat.Albedo = c
		mat.Fuzz = fuzz
		material = mat
	case 2:
		// dielectric
		mat := new(Dielectric)
		mat.Refraction_index = 1.5
		material = mat
	}

	center := Vec3{(a + 0.9*rand.Float32()), height, b + 0.9*rand.Float32()}
	sphere.Center = center
	sphere.Mat = material
	sphere.Radius = height

	return sphere
}

func generateWorld(start int, end int) []Sphere {
	var s []Sphere
	for i := start; i <= end; i++ {
		s = append(s, *generateSphere(float32(start), 2, float32(end)))
	}

	return s
}

func startServer() {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	s := grpc.NewServer()
	pb.RegisterWorldDataServiceServer(s, &server{})
	log.Printf("server running at %v", lis.Addr())
	if err = s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}

func main() {
	interactiveFlag := flag.Bool("it", false, "To run in interactive mode")
	flag.Parse()
	if *interactiveFlag {
		cmdListener()
	} else {
		startServer()
	}
}
