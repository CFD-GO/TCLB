function parse_git_branch_and_add_brackets {
	git branch --no-color 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/\[\1\]/'
}

function parse_git_url {
	git config remote.origin.url 2> /dev/null
}

function parse_git_describe {
	git config --get remote.origin.url | sed -e 's|^git@github.com:|gh |;s|^https://github.com/|gh |;s|\.git$||' 2>/dev/null
#	git describe --tags 2> /dev/null
}

function my.prompt.fun {
	P="$PWD"
	GP=$(git rev-parse --show-toplevel 2>/dev/null)
	if test -z "$GP"
	then
		P=$(echo "$P" | sed -e "s|^$HOME|~|")
		PR="\e[32m\e[1m$P"
	else # GIT
		ORIGIN=$(git config --get remote.origin.url 2>/dev/null)
		ORIGIN=$(echo "$ORIGIN" | sed -E 's|\.git$||;s+^(git@([^:]*):|https://([^/]*)/)+\2\3:+;s|^github.com:|gh:|;s|^([^:]*):(.*)|\1:\\e[1m\2\\e[0m|')
		BRANCH=$(git branch --no-color 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/\1/')
		P=$(echo "$P" | sed -e "s|^$GP||")
		GP=$(echo "$GP" | sed -e "s|^$HOME|~|")
		PR="\e[32m$GP\e[1m$P\e[0m \e[33m$ORIGIN \e[33m[$BRANCH]"
	fi
	echo -ne "$PR"
}

#PS1='\e[0m[\t] \e[32m\h:\w \e[31m$(parse_git_describe) $(parse_git_branch_and_add_brackets)\e[0m\n > '
PS1='\e[0m[\t] \e[32m\h:$(my.prompt.fun)\e[0m\n > '
PS2="+> "
export PS1 PS2
