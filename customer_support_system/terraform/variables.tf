variable "resource_group_name" {
  default = "fastapi-rg"
}

variable "location" {
  default = "eastus"
}

variable "container_app_name" {
  default = "fastapi-app"
}

variable "acr_name" {
  description = "Unique name for Azure Container Registry"
  default     = "fastapiacr123" # must be globally unique
}

variable "docker_image_name" {
  default = "fastapi"
}

variable "docker_image_tag" {
  default = "latest"
}

