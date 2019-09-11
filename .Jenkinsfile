pipeline {
    agent any

    environment {
	PATH = "/usr/lib64/openmpi/bin:/usr/local/cuda-7.5/bin:$PATH"
	LD_LIBRARY_PATH = "/usr/lib64/openmpi/lib:/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH"
    }

    stages {
        stage('Build') {
            steps {
		sh "./tools/install.sh rdep"
		sh "make configure"
		sh "./configure --with-mpi-include=/usr/include/openmpi-x86_64/"
		sh "make d2q9"
            }
        }
        stage('Test') {
            steps {
		sh "git submodule init; git submodule update;"
		sh "tools/tests.sh d2q9"
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying....'
            }
        }
    }
}
