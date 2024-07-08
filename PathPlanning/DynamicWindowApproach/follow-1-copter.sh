#!/bin/bash

# Kill all SITL binaries when exiting
trap "killall -9 arducopter" SIGINT SIGTERM EXIT

ROOTDIR=$PWD
COPTER=$ROOTDIR/build/sitl/bin/arducopter

GCS_IP="127.0.0.1"

# Check if Platform is Native Linux, WSL or Cygwin
unameOut="$(uname -s)"
if [[ "$(expr substr $unameOut 1 5)" == "Linux" ]]; then
    if grep -q Microsoft /proc/version; then
        MCAST_IP_PORT="127.0.0.1:14550"
    else
        MCAST_IP_PORT=""
    fi
elif [[ "$(expr substr $unameOut 1 6)" == "CYGWIN" ]]; then
    MCAST_IP_PORT="0.0.0.0:14550"
fi

BASE_DEFAULTS="$ROOTDIR/Tools/autotest/default_params/copter.parm,$ROOTDIR/Tools/autotest/default_params/airsim-quadX.parm"

[ -x "$COPTER" ] || {
    ./waf configure --board sitl
    ./waf copter
}

# Start up main copter
#$COPTER --model airsim-copter --serial0 udpclient:$GCS_IP:14550 --serial1 mcast:$MCAST_IP_PORT --defaults $BASE_DEFAULTS &
# Start up main copter
$COPTER --model airsim-copter --serial0 udpclient:$GCS_IP:14550 --serial1 mcast:$MCAST_IP_PORT --defaults $BASE_DEFAULTS &

# Start additional copters
#$COPTER --model airsim-copter --serial0 udpclient:$GCS_IP:14551 --serial1 mcast:$MCAST_IP_PORT --instance $i --defaults $BASE_DEFAULTS,follow.parm &

# Set number of extra copters
NCOPTERS="1"

for i in $(seq $NCOPTERS); do
    echo "Starting copter $i"
    mkdir -p copter$i
    SYSID=$(expr $i + 1)
    FOLL_SYSID=$(expr $SYSID - 1)

    # Create default parameter file for the follower
    cat <<EOF > copter$i/follow.parm
SYSID_THISMAV $SYSID
FOLL_ENABLE 1
FOLL_OFS_X -5
FOLL_OFS_TYPE 1
FOLL_SYSID $FOLL_SYSID
FOLL_DIST_MAX 1000
EOF
    pushd copter$i
    $COPTER --model airsim-copter --serial0 udpclient:$GCS_IP:14551 --serial1 mcast:$MCAST_IP_PORT --instance $i --defaults $BASE_DEFAULTS,follow.parm &
    popd
done
wait

