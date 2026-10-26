任务一：
请编写一个Python脚本，按以下要求处理指定路径下的数据：
1. 数据路径：`OpenScholar-DataStore-V3\passages`，目录内为jsonl格式文件，文件中每一行对应一篇文章的一个段落；
2. 数据字段：每条记录包含`text`（段落内容）、`raw_id`（文章唯一ID，同一篇文章的所有段落`raw_id`一致）等字段；
3. 核心需求：
   - 提取每篇文章中`raw_id`第一次出现的段落作为首段落，将所有提取结果保存为一个文件，名为 `paper_first_paragraph.jsonl`；
   - 对提取出的每个首段落，以中文/英文逗号（`,`）、句号（`.`）为分隔符，截取段落的第一句文本，将所有提取结果保存为一个文件，名为 `paper_first_paragraph_first_sentence.jsonl`；
   - 使用`OpenScholar_Retriever`模型，将截取后的第一句文本转换为对应的向量嵌入（embedding），将转换后的所有结果保存为一个文件。


任务二：
请构建一个Python函数，具体要求如下：
1. **函数输入参数**：共3个，分别为**关键词**、**智能体名字**、**输出路径**（输出路径设置默认值为当前路径）。
2. **文件夹创建功能**：函数执行时，自动创建层级目录：`输出路径/智能体名字`。
3. **数据获取要求**：调用`/paper/search/bulk`API接口，根据输入的关键词检索学术文章，接口返回结果仅保留`paperId、title、abstract、year、authors`这5个关键字段。
4. **数据保存要求**：将筛选后的文章信息，保存至上述创建好的文件夹内，输出文件固定命名为`search_paper.json`。


任务三：
续写Jupyter Notebook，新增一个代码单元格，编写一个**支持双功能切换**的Python函数，具体实现要求如下：

函数接收一个目录路径作为参数（示例：`D:\Projects\BotMaker\microbiome_ecology`），具备以下两个可切换执行的核心功能：
1. **向量生成与保存**：读取目录下的`search_paper.jsonl`文件，调用`OpenScholar_Retriever`模型的`encode_texts`方法，将每一条学术文献的`title`字段转换为向量嵌入，并将生成的向量结果保存。
2. **相似检索与结果输出**：基于已生成的标题向量嵌入，调用`multi_index_search`函数匹配最相似的`first sentence`，为每篇文献匹配相似度评分最高的`raw_id`；最终输出一个JSON格式字典，字典的**key**为`索引名.split('.')[0]`，**value**为匹配到的所有`raw_id`。


任务四：
新增一个代码单元格，构建一个 Python 函数，函数接收**两个目录路径参数**：
1. 目标根目录（示例：`D:\Projects\BotMaker\microbiome_ecology`）
2. 原文 passages 存储目录（示例：`D:\Projects\BotMaker\OpenScholar-DataStore-V3\passages`）

函数功能要求：
- 从**目标根目录**下找到所有 `search_results.json` 文件
- 读取每个 `search_results.json`，它是一个 JSON 字典：
  - **key** 是 `OpenScholar-DataStore-V3/passages` 目录下的分块文件名
  - **value** 是需要从对应分块文件中提取的 `raw_id` 列表
- 从**原文 passages 目录**中读取对应分块文件，根据 `raw_id` 提取完整文章信息
- 将提取到的内容**保存到目标根目录下的 passages/ 子目录中**
- 保存结构**严格与原 passages 目录保持一致**，按相同分块规则存储


