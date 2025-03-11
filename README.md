# TACC-Intro-CUDA

Course materials for Introductory CUDA course. 

This repository contains a series of programming labs designed to teach the fundamentals of GPU programming with NVIDIA CUDA.
Each lab builds upon the previous one, introducing progressively more advanced concepts.

## Lab Overview

### [Lab 1: Introduction to CUDA - Array Squaring](lab1/README-lab1.md)
An introduction to the basic structure of CUDA programs through a simple array squaring example.
- Basic CUDA kernel creation
- Host and device memory management
- Thread/block configuration
- Performance measurement

### [Lab 2: CUDA 2D Blocks and Matrix Operations](lab2/README-lab2.md)
Demonstrates how to use 2D thread and block configurations for matrix operations.
- 2D thread and block organization
- Matrix addition and transposition
- Thread indexing in multiple dimensions
- Performance comparison

### [Lab 3: Shared Memory Matrix Addition](lab3/README-lab3.md)
Explores the performance benefits of using shared memory in CUDA.
- Shared memory concepts and declaration
- Thread synchronization with `__syncthreads()`
- Performance comparison between global and shared memory
- Tiling strategies

### [Lab 4: Grace Hopper Memory Allocation Types](lab4/README-lab4.md)
Compares four different memory allocation strategies in CUDA.
- Pageable memory (conventional)
- Pinned memory
- Mapped memory (zero-copy)
- Unified memory
- Performance analysis for different memory types

## Prerequisites

To complete these labs, you'll need:
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- C compiler compatible with CUDA
- NVIDIA HPC SDK (for Fortran versions)

These labs are meant to be run on TACC's Vista cluster on the Grace-Hopper nodes (gh partition).

## Getting Started

Each lab directory contains:
- A detailed README.md file with instructions and discussion topics.
- Source code in both C (.cu) and Fortran (.cuf) versions
- Compilation and execution instructions

Navigate to each lab directory and follow the instructions in the README.md file to compile and run the examples.

## Learning Path

These labs are designed to be completed in order, as each builds upon concepts introduced in previous labs. Start with Lab 1 to learn the fundamentals before progressing to more advanced topics.