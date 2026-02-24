# 1. Resource Group
resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = var.location
}

# 2. Azure Container Registry
resource "azurerm_container_registry" "acr" {
  name                     = var.acr_name
  resource_group_name       = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  sku                      = "Basic"
  admin_enabled            = true
}

# 3. Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "log" {
  name                = "${var.resource_group_name}-law"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

# 4. Container Apps Environment
resource "azurerm_container_app_environment" "env" {
  name                = "${var.resource_group_name}-env"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  log_analytics {
    customer_id = azurerm_log_analytics_workspace.log.customer_id
    shared_key  = azurerm_log_analytics_workspace.log.primary_shared_key
  }
}

# 5. Container App
resource "azurerm_container_app" "app" {
  name                        = var.container_app_name
  container_app_environment_id = azurerm_container_app_environment.env.id
  resource_group_name          = azurerm_resource_group.rg.name
  location                     = azurerm_resource_group.rg.location

  container {
    name   = var.docker_image_name
    image  = "${azurerm_container_registry.acr.login_server}/${var.docker_image_name}:${var.docker_image_tag}"
    cpu    = 0.5
    memory = "1.0Gi"
    ports {
      port     = 80
      protocol = "TCP"
    }
    registry_credentials {
      server   = azurerm_container_registry.acr.login_server
      username = azurerm_container_registry.acr.admin_username
      password = azurerm_container_registry.acr.admin_password
    }
  }

  ingress {
    external_enabled = true
    target_port      = 80
    transport        = "Auto"
  }
}
