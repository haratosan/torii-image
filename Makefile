.PHONY: build install uninstall

NAME = torii-image
INSTALL_DIR = $(HOME)/.local/share/torii/extensions/$(NAME)

build:
	go build -o $(NAME) .

install: build
	@mkdir -p "$(INSTALL_DIR)"
	@cp manifest.json "$(INSTALL_DIR)/"
	@cp $(NAME) "$(INSTALL_DIR)/"
	@echo "Installed $(NAME) to $(INSTALL_DIR)"

uninstall:
	@rm -rf "$(INSTALL_DIR)"
	@echo "Removed $(INSTALL_DIR)"
