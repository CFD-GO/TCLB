pipeline {
    agent any

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
