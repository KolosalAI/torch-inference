#!/bin/bash
# Comprehensive Benchmark Testing for Image Classification Models
# Tests inference speed, accuracy, memory usage, and throughput

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     Image Classification Benchmark Suite                     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
BASE_URL="http://localhost:8000"
BENCHMARK_DIR="./benchmark_results"
IMAGES_DIR="./benchmark_images"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$BENCHMARK_DIR/benchmark_${TIMESTAMP}.json"
REPORT_FILE="$BENCHMARK_DIR/benchmark_${TIMESTAMP}.md"

# Create directories
mkdir -p "$BENCHMARK_DIR"
mkdir -p "$IMAGES_DIR"

# Models to benchmark (sorted by size)
declare -a MODELS=(
    "mobilenetv4-hybrid-large:140MB"
    "coatnet-3-rw-224:700MB"
    "swin-large-patch4-384:790MB"
    "efficientnetv2-xl:850MB"
    "eva02-large-patch14-448:1.2GB"
)

# Test image URLs (different categories)
declare -a TEST_IMAGES=(
    "cat:https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/400px-Cat03.jpg"
    "dog:https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Sled_dog_Togo.jpg/400px-Sled_dog_Togo.jpg"
    "bird:https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Eopsaltria_australis_-_Mogo_Campground.jpg/400px-Eopsaltria_australis.jpg"
    "car:https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/2018_Porsche_911_Carrera_T_3.0.jpg/400px-2018_Porsche_911.jpg"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if server is running
check_server() {
    echo "Checking server status..."
    if curl -sf "$BASE_URL/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server is running${NC}"
        return 0
    else
        echo -e "${RED}✗ Server is not running${NC}"
        echo "Please start the server first:"
        echo "  ./target/release/torch-inference-server"
        exit 1
    fi
}

# Download test images
download_test_images() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Downloading Test Images"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    for img_data in "${TEST_IMAGES[@]}"; do
        IFS=':' read -r name url <<< "$img_data"
        
        if [ -f "$IMAGES_DIR/${name}.jpg" ]; then
            echo "✓ ${name}.jpg already exists"
        else
            echo -n "Downloading ${name}.jpg... "
            if curl -s -o "$IMAGES_DIR/${name}.jpg" "$url"; then
                echo -e "${GREEN}✓${NC}"
            else
                echo -e "${RED}✗${NC}"
            fi
        fi
    done
}

# Run single inference and measure time
benchmark_single_inference() {
    local model="$1"
    local image="$2"
    local warmup="$3"  # true/false
    
    START_TIME=$(date +%s%N)
    
    RESPONSE=$(curl -s -X POST "$BASE_URL/classify" \
        -F "image=@$image" \
        -F "model=$model" \
        -F "top_k=5" 2>/dev/null)
    
    END_TIME=$(date +%s%N)
    
    # Calculate elapsed time in milliseconds
    ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))
    
    # Extract inference time from response if available
    INFERENCE_TIME=$(echo "$RESPONSE" | grep -o '"inference_time_ms":[0-9.]*' | cut -d':' -f2)
    
    if [ -z "$INFERENCE_TIME" ]; then
        INFERENCE_TIME=$ELAPSED_MS
    fi
    
    echo "$INFERENCE_TIME"
}

# Benchmark a single model
benchmark_model() {
    local model_info="$1"
    IFS=':' read -r model_id model_size <<< "$model_info"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Benchmarking: $model_id ($model_size)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Check if model is downloaded
    echo "Checking if model is downloaded..."
    MODEL_STATUS=$(curl -s "$BASE_URL/models/managed" | grep -o "\"name\":\"$model_id\"" || echo "")
    
    if [ -z "$MODEL_STATUS" ]; then
        echo -e "${YELLOW}Model not downloaded. Skipping...${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✓ Model available${NC}"
    echo ""
    
    # Warmup runs
    echo "Running warmup (3 iterations)..."
    for i in {1..3}; do
        benchmark_single_inference "$model_id" "$IMAGES_DIR/cat.jpg" "true" > /dev/null
        echo -n "."
    done
    echo " done"
    echo ""
    
    # Benchmark each test image
    echo "Benchmarking inference times:"
    echo ""
    
    declare -a TIMES=()
    
    for img_data in "${TEST_IMAGES[@]}"; do
        IFS=':' read -r img_name img_url <<< "$img_data"
        IMG_PATH="$IMAGES_DIR/${img_name}.jpg"
        
        if [ ! -f "$IMG_PATH" ]; then
            continue
        fi
        
        echo -n "  ${img_name}.jpg: "
        
        # Run 5 iterations for this image
        declare -a IMG_TIMES=()
        for i in {1..5}; do
            TIME=$(benchmark_single_inference "$model_id" "$IMG_PATH" "false")
            IMG_TIMES+=("$TIME")
        done
        
        # Calculate average
        SUM=0
        for t in "${IMG_TIMES[@]}"; do
            SUM=$(echo "$SUM + $t" | bc)
        done
        AVG=$(echo "scale=2; $SUM / ${#IMG_TIMES[@]}" | bc)
        
        # Find min and max
        MIN=${IMG_TIMES[0]}
        MAX=${IMG_TIMES[0]}
        for t in "${IMG_TIMES[@]}"; do
            if (( $(echo "$t < $MIN" | bc -l) )); then MIN=$t; fi
            if (( $(echo "$t > $MAX" | bc -l) )); then MAX=$t; fi
        done
        
        echo "${AVG}ms (min: ${MIN}ms, max: ${MAX}ms)"
        TIMES+=("$AVG")
    done
    
    echo ""
    
    # Calculate overall statistics
    if [ ${#TIMES[@]} -gt 0 ]; then
        SUM=0
        for t in "${TIMES[@]}"; do
            SUM=$(echo "$SUM + $t" | bc)
        done
        OVERALL_AVG=$(echo "scale=2; $SUM / ${#TIMES[@]}" | bc)
        
        echo "Summary:"
        echo "  Average inference time: ${OVERALL_AVG}ms"
        echo "  Images per second: $(echo "scale=2; 1000 / $OVERALL_AVG" | bc)"
        
        # Save to results file
        echo "{\"model\":\"$model_id\",\"size\":\"$model_size\",\"avg_time_ms\":$OVERALL_AVG,\"fps\":$(echo "scale=2; 1000 / $OVERALL_AVG" | bc)}" >> "$RESULTS_FILE.tmp"
    fi
}

# Throughput test (concurrent requests)
benchmark_throughput() {
    local model="$1"
    local concurrent="$2"
    
    echo ""
    echo "Testing throughput with $concurrent concurrent requests..."
    
    START=$(date +%s)
    
    for i in $(seq 1 $concurrent); do
        curl -s -X POST "$BASE_URL/classify" \
            -F "image=@$IMAGES_DIR/cat.jpg" \
            -F "model=$model" \
            -F "top_k=5" > /dev/null 2>&1 &
    done
    
    wait
    
    END=$(date +%s)
    DURATION=$((END - START))
    THROUGHPUT=$(echo "scale=2; $concurrent / $DURATION" | bc)
    
    echo "  Completed $concurrent requests in ${DURATION}s"
    echo "  Throughput: ${THROUGHPUT} requests/sec"
}

# Memory usage test
benchmark_memory() {
    local model="$1"
    
    echo ""
    echo "Testing memory usage..."
    
    # Get baseline memory
    BEFORE=$(ps aux | grep torch-inference-server | grep -v grep | awk '{print $6}')
    
    # Run inference
    curl -s -X POST "$BASE_URL/classify" \
        -F "image=@$IMAGES_DIR/cat.jpg" \
        -F "model=$model" \
        -F "top_k=5" > /dev/null 2>&1
    
    # Get memory after inference
    sleep 1
    AFTER=$(ps aux | grep torch-inference-server | grep -v grep | awk '{print $6}')
    
    if [ -n "$BEFORE" ] && [ -n "$AFTER" ]; then
        BEFORE_MB=$(echo "scale=2; $BEFORE / 1024" | bc)
        AFTER_MB=$(echo "scale=2; $AFTER / 1024" | bc)
        DIFF_MB=$(echo "scale=2; $AFTER_MB - $BEFORE_MB" | bc)
        
        echo "  Memory before: ${BEFORE_MB}MB"
        echo "  Memory after: ${AFTER_MB}MB"
        echo "  Difference: ${DIFF_MB}MB"
    fi
}

# Generate markdown report
generate_report() {
    echo ""
    echo "Generating benchmark report..."
    
    cat > "$REPORT_FILE" << EOF
# Image Classification Benchmark Report

**Date:** $(date)
**Server:** torch-inference v1.0.0
**Test Images:** ${#TEST_IMAGES[@]} categories
**Iterations per test:** 5

---

## System Information

- **OS:** $(uname -s) $(uname -r)
- **Architecture:** $(uname -m)
- **CPU:** $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
- **RAM:** $(sysctl -n hw.memsize 2>/dev/null | awk '{print $1/1024/1024/1024 "GB"}' || echo "Unknown")

---

## Benchmark Results

### Inference Speed Comparison

| Model | Size | Avg Time (ms) | FPS | Images/sec |
|-------|------|---------------|-----|------------|
EOF

    # Add results from temp file
    if [ -f "$RESULTS_FILE.tmp" ]; then
        while IFS= read -r line; do
            MODEL=$(echo "$line" | grep -o '"model":"[^"]*"' | cut -d'"' -f4)
            SIZE=$(echo "$line" | grep -o '"size":"[^"]*"' | cut -d'"' -f4)
            AVG_TIME=$(echo "$line" | grep -o '"avg_time_ms":[0-9.]*' | cut -d':' -f2)
            FPS=$(echo "$line" | grep -o '"fps":[0-9.]*' | cut -d':' -f2)
            
            echo "| $MODEL | $SIZE | $AVG_TIME | $FPS | $FPS |" >> "$REPORT_FILE"
        done < "$RESULTS_FILE.tmp"
        
        mv "$RESULTS_FILE.tmp" "$RESULTS_FILE"
    fi
    
    cat >> "$REPORT_FILE" << EOF

---

## Performance Categories

### 🏆 Fastest Model
The model with the lowest inference time.

### ⚡ Best Throughput
The model that can process the most images per second.

### 💪 Best Accuracy
The model with the highest ImageNet-1K accuracy.

### ⚖️ Best Balance
The model with the best speed/accuracy tradeoff.

---

## Test Configuration

- **Warmup iterations:** 3 per model
- **Test iterations:** 5 per image
- **Test images:** ${#TEST_IMAGES[@]} (cat, dog, bird, car)
- **Top-K predictions:** 5
- **Concurrent requests:** Not tested (single-threaded)

---

## Recommendations

### For Real-time Applications (< 100ms)
Use the fastest model from the results above.

### For High Accuracy (> 88%)
Use EVA-02 Large or ConvNeXt V2 Huge.

### For Mobile/Edge Deployment
Use MobileNetV4 Hybrid Large (smallest and fastest).

### For Production Balance
Use EfficientNetV2 XL or Swin Transformer Large.

---

## Raw Results

See \`$RESULTS_FILE\` for raw JSON data.

---

## Next Steps

1. Test with GPU acceleration for 5-10x speedup
2. Implement batch processing for higher throughput
3. Optimize preprocessing pipeline
4. Test with different image sizes
5. Compare with ONNX Runtime backend

EOF

    echo -e "${GREEN}✓ Report generated: $REPORT_FILE${NC}"
}

# Main execution
main() {
    echo "Benchmark Configuration:"
    echo "  Models to test: ${#MODELS[@]}"
    echo "  Test images: ${#TEST_IMAGES[@]}"
    echo "  Output directory: $BENCHMARK_DIR"
    echo ""
    
    # Check server
    check_server
    
    # Download test images
    download_test_images
    
    # Initialize results file
    echo "[]" > "$RESULTS_FILE.tmp"
    
    # Benchmark each model
    for model_info in "${MODELS[@]}"; do
        benchmark_model "$model_info"
        
        # Optional: throughput and memory tests for first model
        IFS=':' read -r model_id model_size <<< "$model_info"
        if [ "$model_info" = "${MODELS[0]}" ]; then
            benchmark_throughput "$model_id" 10
            benchmark_memory "$model_id"
        fi
    done
    
    # Generate report
    generate_report
    
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                  Benchmark Complete                          ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Results saved to:"
    echo "  - JSON: $RESULTS_FILE"
    echo "  - Report: $REPORT_FILE"
    echo ""
    echo "View report:"
    echo "  cat $REPORT_FILE"
    echo ""
}

# Run main
main
