main: src/main.cpp
	g++ -std=gnu++17 -Wall -Wextra -DNDEBUG -mtune=native -march=native -Ofast -I . -o src/main ./src/main.cpp

.PHONY: clean

clean:
	rm src/main