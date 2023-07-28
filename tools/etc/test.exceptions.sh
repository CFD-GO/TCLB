#!/bin/bash

case "$MODEL" in
d3q27_PSM_NEBB)
	case "$TEST" in
	single_inlet_sp)
		CAN_FAIL=true
		;;
	esac
	;;
d3q27_pf_velocity*)
	CSV_DISCARD="$CSV_DISCARD,InObj"
	;;
esac
