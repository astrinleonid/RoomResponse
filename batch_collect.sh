#!/bin/bash

# Batch Room Response Data Collection Script
# Collects scenarios 0.1 through 0.5 with 200 measurements each

echo "Room Response Batch Data Collection"
echo "=================================="
echo "Scenarios: 0.1, 0.2, 0.3, 0.4, 0.5"
echo "Measurements per scenario: 200"
echo "Measurement interval: 1.0 seconds"
echo "Total estimated time: ~100 minutes"
echo "=================================="

# Confirm start
read -p "Proceed with batch collection? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Batch collection cancelled."
    exit 0
fi

# Scenario parameters
scenarios=("0.1" "0.2" "0.3" "0.4" "0.5")
num_measurements=2
measurement_interval=1.0

# Track results
completed=()
failed=()

# Start time
start_time=$(date +%s)

# Run each scenario
for i in "${!scenarios[@]}"; do
    scenario="${scenarios[$i]}"
    scenario_num=$((i + 1))
    total_scenarios=${#scenarios[@]}

    echo
    echo "🚀 Starting scenario $scenario_num/$total_scenarios: $scenario"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"

    # Run the collection
    if python3 collect_scenario.py \
        --quiet \
        --scenario-number "$scenario" \
        --description "Automated collection scenario $scenario" \
        --num-measurements "$num_measurements" \
        --measurement-interval "$measurement_interval"; then

        echo "✅ Scenario $scenario completed successfully!"
        completed+=("$scenario")
    else
        echo "❌ Scenario $scenario failed!"
        failed+=("$scenario")
    fi

    # Ask to continue (except for last scenario)
    if [[ $scenario_num -lt $total_scenarios ]]; then
        remaining_scenarios=("${scenarios[@]:$scenario_num}")
        echo
        echo "Scenario $scenario finished."
        echo "Remaining scenarios: ${remaining_scenarios[*]}"
        echo
        read -p "Continue to next scenario? (y/n/s=skip remaining): " -n 1 -r
        echo

        case $REPLY in
            [Nn])
                echo "Batch collection stopped by user."
                break
                ;;
            [Ss])
                echo "Skipping remaining scenarios."
                break
                ;;
            *)
                echo "Continuing in 5 seconds..."
                sleep 5
                ;;
        esac
    fi
done

# Calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))
duration_minutes=$((duration / 60))

# Print summary
echo
echo "==============================================="
echo "BATCH COLLECTION SUMMARY"
echo "==============================================="
echo "Total duration: $duration_minutes minutes"
echo "Completed scenarios: ${#completed[@]}"
if [[ ${#completed[@]} -gt 0 ]]; then
    echo "  ✅ ${completed[*]}"
fi

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "Failed scenarios: ${#failed[@]}"
    echo "  ❌ ${failed[*]}"
fi

total_measurements=$((${#completed[@]} * num_measurements))
echo "Total measurements collected: ~$total_measurements"
echo "==============================================="

# Exit with appropriate code
if [[ ${#failed[@]} -gt 0 ]]; then
    echo
    echo "⚠️ Some scenarios failed. Check the output above for details."
    exit 1
elif [[ ${#completed[@]} -gt 0 ]]; then
    echo
    echo "🎉 Batch collection completed successfully!"
    exit 0
else
    echo
    echo "❌ No scenarios completed successfully."
    exit 1
fi