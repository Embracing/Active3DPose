#!/usr/bin/env bash

PROCNAMES=(ray python python3 AnimalParsing)

for procname in "${PROCNAMES[@]}"; do
	while true; do
		pids=()
		while IFS='' read -r pid; do
			if pgrep -P "${pid}" &>/dev/null; then
				continue
			fi
			pids+=("${pid}")
		done < <(pgrep -P 1 -u "${USER}" "${procname}")
		if [[ "${#pids}" -gt 0 ]]; then
			ps -o "pid,user,command" -p "${pids[@]}"
			kill -9 "${pids[@]}" 2>/dev/null
			sleep 10
			echo
		else
			break
		fi
	done
done
