# Check if running as root, if not use sudo
if [ "$EUID" -eq 0 ]; then
    APT_CMD="apt-get"
    INSTALL_CMD="apt install"
else
    APT_CMD="sudo apt-get"
    INSTALL_CMD="sudo apt install"
fi

$APT_CMD update
$INSTALL_CMD -y build-essential libtool autoconf automake libnuma-dev unzip pkg-config librdmacm-dev rdma-core make cmake python3-pip

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/../ &> /dev/null && pwd )"

rm -rf zeromq-4.1.4.tar.gz zeromq-4.1.4

# Download with timeout and retry options
echo "Downloading zeromq-4.1.4.tar.gz..."
wget --timeout=30 --tries=3 --retry-connrefused https://raw.githubusercontent.com/mli/deps/master/build/zeromq-4.1.4.tar.gz || {
    echo "Error: Failed to download zeromq. Trying alternative method..."
    curl -L --max-time 60 --retry 3 -o zeromq-4.1.4.tar.gz https://raw.githubusercontent.com/mli/deps/master/build/zeromq-4.1.4.tar.gz || {
        echo "Error: Both wget and curl failed. Please check your network connection."
        exit 1
    }
}
tar --no-same-owner -zxf zeromq-4.1.4.tar.gz
pushd zeromq-4.1.4 || exit
export CFLAGS=-fPIC
export CXXFLAGS=-fPIC

./configure -prefix=${THIS_DIR}/deps/ --with-libsodium=no --with-libgssapi_krb5=no
make -j
make install
popd || exit

rm -rf zeromq-4.1.4.tar.gz zeromq-4.1.4
