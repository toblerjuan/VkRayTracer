CFLAGS = -g -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

TriangleTest: main.cpp lib/tiny_obj_loader.h
	export NV_ALLOW_RAYTRACING_VALIDATION=1
	g++ $(CFLAGS) -o build/TriangleTest main.cpp $(LDFLAGS)

.PHONY: test clean

test: TriangleTest
	./build/TriangleTest

clean:
	rm -f build/*
