我需要为 `src/` 目录下的 BotMaker pipeline 开发一个**上游模块**，功能是：根据用户输入的**研究主题上下文描述**，自动生成一个**多专家团队**，并严格按照下面给定的 JSON Schema 返回结果。

生成多专家团队的输入参数包括：
1. 研究主题的上下文描述
2. 专家数量
3. 专家之间的差异度（1–10 分，数值越大，专家之间的差异性越强，数值越小，专家之间的差异性越小）

请参考 `ar_science` 项目的核心思想，设计一套**可控的专家差异化生成流程**，目标是生成**跨领域、高多样性**的专家团队，让不同领域的专家围绕科学问题展开讨论，从而产生更具创造性的思路。

你需要帮我设计一个完整的实现方法：
- 借鉴 `ar_science` 的思想
- 但不能完全照搬，需要具备创新性
- 能够通过流程控制专家差异度
- 最终生成适配研究主题的专家团队

JSON Schema 如下：
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ExpertTeamGeneration",
  "description": "Schema for defining a team of experts with structured roles and descriptions",
  "type": "object",
  "required": ["team"],
  "properties": {
    "team": {
      "type": "object",
      "title": "Team",
      "description": "A collection of expert members mapped by unique keys",
      "additionalProperties": {
        "type": "object",
        "title": "Expert",
        "description": "Individual expert member details",
        "required": ["type", "expert_description"],
        "properties": {
          "type": {
            "type": "string",
            "title": "ExpertType",
            "description": "Role or category of the expert",
            "minLength": 1
          },
          "expert_description": {
            "type": "string",
            "title": "ExpertDescription",
            "description": "Detailed expertise, responsibility, or background",
            "minLength": 5
          }
        },
        "additionalProperties": false
      }
    }
  },
  "additionalProperties": false
}
```