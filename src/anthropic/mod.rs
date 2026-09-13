pub mod client;

#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct Model {
   pub id: String,
   pub display_name: String,
   pub created_at: String,
   #[serde(rename = "type")]
   pub kind: String,
}

#[derive(serde::Deserialize, serde::Serialize)]
pub struct Catalog {
   pub data: Vec<Model>,
   pub has_more: bool,
   pub first_id: Option<String>,
   pub last_id: Option<String>,
}
