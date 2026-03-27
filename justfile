default: list

# List available commands
list:
    @just --list --unsorted

test:
    @cd test && make

bench-all:
    @cd bench && make run

bench-micro:
    @cd bench && make run-micro

bench-composite:
    @cd bench && make run-composite

bench-mnist:
    @cd bench && make run-mnist
