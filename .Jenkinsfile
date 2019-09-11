pipeline {
    agent any

    environment {
	PATH = "/usr/local/cuda-7.5/bin:$PATH"
	LD_LIBRARY_PATH = "/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH"
	PATH = "/usr/lib64/openmpi/bin:$PATH"
	LD_LIBRARY_PATH = "/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH"
    }

    stages {
        stage('Build') {
            steps {
		sh "make configure"
		sh "./configure"
		sh "make d2q9"
            }
        }
        stage('Test') {
            steps {
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
